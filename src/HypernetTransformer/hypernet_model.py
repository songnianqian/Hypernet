# MIT License
#
# HyperNet GPT-2: Complete Model
# 
# Includes ALL features:
# - 20-layer hardcoded architecture (10 standard + 10 hypernet)
# - Clustered hypernetwork with M=128 centroids
# - Optional RL for centroid selection
# - Logic injection
# - All 5 architectural changes
# - Compatible with orthogonal projection optimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Optional, Tuple, List, Dict
import math
import json
from pathlib import Path


class ReasoningLogicInjector:
    """Injects reasoning features into token embeddings"""
    
    def __init__(self, tokenizer, reasoning_json_path: str = "reasoning_logic.json"):
        self.tokenizer = tokenizer
        self.feature_dim = 4
        
        if Path(reasoning_json_path).exists():
            with open(reasoning_json_path, 'r') as f:
                self.logic_map = json.load(f)
        else:
            self.logic_map = {
                "numerical": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                "operators": ["+", "-", "*", "/", "=", ">", "<"],
                "logical": ["if", "then", "else", "and", "or", "not", "therefore"],
                "delimiters": ["(", ")", "[", "]", "{", "}", ";", ","]
            }
        
        self.token_features = self._build_token_features()
        print(f"✅ Logic Injector: {len(self.token_features)} categorized tokens")
    
    def _build_token_features(self) -> Dict[int, torch.Tensor]:
        token_features = {}
        categories = [("numerical", 0), ("operators", 1), ("logical", 2), ("delimiters", 3)]
        
        for category_name, feature_idx in categories:
            tokens = self.logic_map.get(category_name, [])
            for token_str in tokens:
                token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                for token_id in token_ids:
                    if token_id not in token_features:
                        token_features[token_id] = torch.zeros(self.feature_dim)
                    token_features[token_id][feature_idx] = 1.0
        
        return token_features
    
    def get_reasoning_features(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        features = torch.zeros(batch_size, seq_len, self.feature_dim, device=device)
        
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = input_ids[b, s].item()
                if token_id in self.token_features:
                    features[b, s] = self.token_features[token_id].to(device)
        
        return features


class ClusteredHypernetwork(nn.Module):
    """
    Clustered hypernetwork with optional RL for centroid selection.
    
    Features:
    - M=128 learnable centroids
    - Cosine similarity matching
    - Shared perceptron across layers
    - Optional RL for discrete centroid selection
    """
    
    def __init__(
        self,
        config: GPT2Config,
        num_centroids: int = 128,
        memory_window: Optional[int] = None,
        normalize: bool = True,
        use_rl: bool = False,
        rl_method: str = 'actor_critic',
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = config.n_embd
        self.num_centroids = num_centroids
        self.memory_window = memory_window
        self.normalize = normalize
        self.use_rl = use_rl
        self.rl_method = rl_method if use_rl else None
        self.entropy_weight = entropy_weight
        
        # Learnable centroids
        self.centroids = nn.Parameter(torch.randn(num_centroids, self.hidden_size))
        nn.init.xavier_uniform_(self.centroids)
        
        # Shared perceptron
        self.shared_perceptron = nn.Sequential(
            nn.Linear(num_centroids, num_centroids * 2),
            nn.ReLU(),
            nn.Linear(num_centroids * 2, self.hidden_size),
        )
        
        # Router perceptron: produces a 128-d "reasoning signal" per token (kept as an array, not collapsed to hidden_size)
        self.router_perceptron = nn.Sequential(
            nn.Linear(num_centroids, num_centroids),
            nn.ReLU(),
        )
        
        # RL components (optional)
        if self.use_rl:
            self.value_network = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, 1),
            )
            
            self.policy_network = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, num_centroids),
            )
            
            self.trajectory_buffer = {
                'states': [], 'actions': []
            }
        mode_str = f"RL ({rl_method})" if use_rl else "Standard"
        print(f"🧠 Clustered Hypernetwork: {num_centroids} centroids, {mode_str}")
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    
    def create_rope_position_encoding(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        half_dim = self.hidden_size // 2
        positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        inv_freq = torch.exp(
            torch.arange(0, half_dim, device=device, dtype=dtype) 
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / half_dim)
        )
        
        angles = positions * inv_freq.unsqueeze(0)
        sin, cos = torch.sin(angles), torch.cos(angles)
        
        sin_pos = torch.zeros(seq_len, self.hidden_size, device=device, dtype=dtype)
        cos_pos = torch.zeros(seq_len, self.hidden_size, device=device, dtype=dtype)
        sin_pos[:, ::2] = sin_pos[:, 1::2] = sin
        cos_pos[:, ::2] = cos_pos[:, 1::2] = cos
        return sin_pos, cos_pos
    
    def build_Y_from_X(self, X: torch.Tensor):
        """Build RoPE-like features Y plus its sin/cos components.

        Returns:
            Y:      [B,S,E] interleaved sin/cos features (as before)
            sX:     [B,S,E] sin(X_rot)
            cX:     [B,S,E] cos(X_rot)
        """
        b, s, e = X.shape
        sin_pos, cos_pos = self.create_rope_position_encoding(s, X.device, X.dtype)
        
        X_rot = X * cos_pos.unsqueeze(0) + self.rotate_half(X) * sin_pos.unsqueeze(0)
        sX, cX = torch.sin(X_rot), torch.cos(X_rot)
        
        Y = torch.empty_like(X_rot)
        Y[..., ::2], Y[..., 1::2] = sX[..., ::2], cX[..., ::2]
        return Y, sX, cX
    
    def get_centroid_distribution(self, Y_norm: torch.Tensor, training: bool = True):
        b, s, _ = Y_norm.shape
        centroids_norm = F.normalize(self.centroids, p=2, dim=-1)
        similarity = torch.matmul(Y_norm, centroids_norm.t())
        
        if self.use_rl and training:
            Y_flat = Y_norm.reshape(b * s, self.hidden_size)
            policy_logits = self.policy_network(Y_flat).reshape(b, s, self.num_centroids)
            logits = similarity + policy_logits
        else:
            logits = similarity
        
        action_probs = F.softmax(logits, dim=-1)
        
        if training and self.use_rl:
            selected_centroids = torch.multinomial(
                action_probs.reshape(-1, self.num_centroids), num_samples=1
            ).reshape(b, s)
            
            log_probs = torch.log(action_probs + 1e-10)
            selected_log_probs = log_probs.gather(
                dim=2, index=selected_centroids.unsqueeze(-1)
            ).squeeze(-1)
            
            return action_probs, selected_centroids, selected_log_probs
        else:
            selected_centroids = torch.argmax(action_probs, dim=-1)
            log_probs = torch.zeros(b, s, device=Y_norm.device)
            return action_probs, selected_centroids, log_probs
    
    def _compute_action_stats(self, Y_norm: torch.Tensor, actions: torch.Tensor, training: bool = True):
        """
        def _compute_action_stats(self, Y_norm: torch.Tensor, actions: torch.Tensor, training: bool = True):
    
        RL helper: recompute policy/value quantities from stored states (Y_norm) and stored actions.

        Why we recompute:
        - During the forward pass we DO NOT want to store autograd graphs inside the trajectory buffer,
        because that would explode memory (graph per batch × seq × step).
        - So we store only:
            * state = Y_norm.detach()
            * action = sampled centroid index (int)
        and later re-run the lightweight policy/value networks on the saved states to build a fresh graph.

        What this function returns and why each term is needed:

        1) action_probs  = π(a | s)
        - The policy distribution over centroids for each token state.
        - Useful for diagnostics (is routing peaked or diffuse?) and for computing entropy.

        2) selected_log_probs = log π(a_t | s_t)
        - This is the core quantity for policy-gradient learning.
        - REINFORCE / Actor-Critic optimize:
                L_policy = - E[ advantage_t * log π(a_t | s_t) ]
            So we need the log-probability of the action that was actually taken.

        3) entropy = H(π(.|s))
        - Entropy regularization encourages exploration / avoids premature collapse to one centroid.
        - We add it as:
                L_entropy = -β * H
            (negative sign because we MINIMIZE loss; maximizing entropy helps exploration.)

        4) values = V(s)   (optional; only used for actor-critic)
        - The value network predicts expected return from a state.
        - Used as a learned baseline to reduce variance:
                advantage_t = return_t - V(s_t)
        - Also trained with a regression loss:
                L_value = 0.5 * (return_t - V(s_t))^2

        Notes:
        - 'training' can control whether we sample actions (exploration) or use argmax (greedy).
        In this function, actions are already provided, so we only evaluate π and V consistently.
        """

        b, s, _ = Y_norm.shape
        centroids_norm = F.normalize(self.centroids, p=2, dim=-1)
        similarity = torch.matmul(Y_norm, centroids_norm.t())

        if self.use_rl and training:
            Y_flat = Y_norm.reshape(b * s, self.hidden_size)
            policy_logits = self.policy_network(Y_flat).reshape(b, s, self.num_centroids)
            logits = similarity + policy_logits
        else:
            logits = similarity

        action_probs = F.softmax(logits, dim=-1)

        # Log prob of the actions that were actually taken
        log_probs = torch.log(action_probs + 1e-10)
        selected_log_probs = log_probs.gather(dim=2, index=actions.unsqueeze(-1)).squeeze(-1)

        entropy = -(action_probs * log_probs).sum(dim=-1)

        values = None
        if self.use_rl and training and hasattr(self, "value_network") and self.value_network is not None:
            Y_flat = Y_norm.reshape(b * s, self.hidden_size)
            values = self.value_network(Y_flat).squeeze(-1).reshape(b, s)

        return action_probs, selected_log_probs, entropy, values

    def forward(self, layer_inputs: List[torch.Tensor], training: bool = True):
        if layer_inputs is None or len(layer_inputs) == 0:
            return None, {}
        
        if self.memory_window and len(layer_inputs) > self.memory_window:
            layer_inputs = layer_inputs[-self.memory_window:]
        
        X0 = layer_inputs[0]
        b, s, _ = X0.shape
        
        Y_aggregated = torch.zeros(b, s, self.hidden_size, device=X0.device, dtype=X0.dtype)
        sX_aggregated = torch.zeros_like(Y_aggregated)
        cX_aggregated = torch.zeros_like(Y_aggregated)
        for X in layer_inputs:
            Y, sX, cX = self.build_Y_from_X(X)
            Y_aggregated += Y
            sX_aggregated += sX
            cX_aggregated += cX
        Y_aggregated = Y_aggregated / len(layer_inputs)
        sX_aggregated = sX_aggregated / len(layer_inputs)
        cX_aggregated = cX_aggregated / len(layer_inputs)
        
        Y_norm = F.normalize(Y_aggregated, p=2, dim=-1)
        
        action_probs, selected_centroids, log_probs = self.get_centroid_distribution(
            Y_norm, training=training
        )
        
        values = None
        if self.use_rl and training:
            Y_flat = Y_norm.reshape(b * s, self.hidden_size)
            values = self.value_network(Y_flat).squeeze(-1).reshape(b, s)
        
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1)
        
        # Build a richer routing signal for the FFN:
        #  - router_out:  [B,S,M]   learned transform of main centroid mixture
        #  - dsin_probs:  [B,S,M]   centroid mixture derived from d/dx sin = cos
        #  - dcos_probs:  [B,S,M]   centroid mixture derived from d/dx cos = -sin
        
        # Main router outputs (128-d array per token)
        centroid_weights_flat = action_probs.reshape(b * s, self.num_centroids)
        router_out = self.router_perceptron(centroid_weights_flat).reshape(b, s, self.num_centroids)
        
        # Derivative features (computed from aggregated sin/cos components)
        dsin = cX_aggregated
        dcos = -sX_aggregated
        dsin_norm = F.normalize(dsin, p=2, dim=-1)
        dcos_norm = F.normalize(dcos, p=2, dim=-1)
        centroids_norm = F.normalize(self.centroids, p=2, dim=-1)
        sim_dsin = torch.matmul(dsin_norm, centroids_norm.t())
        sim_dcos = torch.matmul(dcos_norm, centroids_norm.t())
        dsin_probs = F.softmax(sim_dsin, dim=-1)
        dcos_probs = F.softmax(sim_dcos, dim=-1)
        
        # Final routing features used by HyperNetBlock for FFN injection: [B,S,3M]
        reasoning_vector = torch.cat([router_out, dsin_probs, dcos_probs], dim=-1)
        rl_info = {
            'action_probs': action_probs,
            'selected_centroids': selected_centroids,
            'log_probs': log_probs,
            'values': values,
            'entropy': entropy,
            'states': Y_norm,
        }
        
        if training and self.use_rl:
            # Store only states/actions (detached) to avoid holding large graphs.
            # We will recompute log_probs/values/entropy during RL update.
            self.trajectory_buffer['states'].append(Y_norm.detach())
            self.trajectory_buffer['actions'].append(selected_centroids.detach())
        
        return reasoning_vector, rl_info
    
    def compute_rl_loss(self, rewards: torch.Tensor):
        """Compute the RL loss for the HyperNet routing policy.

            High-level idea:
            - HyperNet chooses a centroid index (action) per token position.
            - We treat next-token prediction improvement as a reward signal.
            - We update the policy so that centroid choices that lead to higher reward become more likely.

            Two supported training modes:

            (A) REINFORCE (Monte Carlo policy gradient, no value network baseline)
                advantage_t = returns_t  (optionally normalized)
                L_policy    = - mean( advantage_t * log π(a_t|s_t) )
                Optional: entropy regularization to keep exploration:
                L_entropy   = -β * mean( H(π(.|s)) )
                Total:
                L = L_policy + L_entropy

            (B) Actor-Critic (learned value baseline to reduce variance)
                values_t    = V(s_t)
                advantage_t = returns_t - values_t.detach()
                L_policy    = - mean( advantage_t * log π(a_t|s_t) )
                L_value     = 0.5 * mean( (returns_t - values_t)^2 )
                L_entropy   = -β * mean( H(π(.|s)) )
                Total:
                L = L_policy + c_v * L_value + L_entropy

            Implementation detail:
            - We store only (state, action) in the trajectory buffer during forward passes.
            - During RL update we recompute π and V from stored states so gradients flow properly,
            without keeping huge autograd graphs in memory.
        """
        if not self.use_rl or len(self.trajectory_buffer.get('states', [])) == 0:
            return torch.tensor(0.0, device=rewards.device if isinstance(rewards, torch.Tensor) else None), {}

        # Stack detached states/actions: [T, B, S, H] and [T, B, S]
        states = torch.stack(self.trajectory_buffer['states'], dim=0).to(rewards.device)
        actions = torch.stack(self.trajectory_buffer['actions'], dim=0).to(rewards.device)

        T, B, S, H = states.shape

        # Broadcast rewards to [T, B, S]
        if rewards.dim() == 2:
            rewards_t = rewards.unsqueeze(0).expand(T, -1, -1)
        elif rewards.dim() == 3:
            rewards_t = rewards
        else:
            raise ValueError(f"rewards must have shape [B,S] or [T,B,S], got {tuple(rewards.shape)}")

        # Recompute log-probs / entropy / values WITH gradient graphs
        # Flatten time into batch for efficiency
        states_flat = states.reshape(T * B, S, H)
        actions_flat = actions.reshape(T * B, S)

        _, selected_log_probs, entropy, values = self._compute_action_stats(
            states_flat, actions_flat, training=True
        )

        # Reshape back to [T, B, S]
        selected_log_probs = selected_log_probs.reshape(T, B, S)
        entropy = entropy.reshape(T, B, S)
        if values is not None:
            values = values.reshape(T, B, S)

        if self.rl_method == 'reinforce':
            # Optional baseline: subtract mean reward per sequence to reduce variance
            baseline = rewards_t.mean(dim=-1, keepdim=True)
            advantages = rewards_t - baseline
            policy_loss = -(selected_log_probs * advantages.detach()).mean()
            value_loss = torch.tensor(0.0, device=policy_loss.device)
        else:
            # actor_critic (default)
            if values is None:
                raise RuntimeError("actor_critic selected but value_network is missing")
            advantages = rewards_t - values
            policy_loss = -(selected_log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(values, rewards_t)

        entropy_loss = -self.entropy_weight * entropy.mean()
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss

        info = {
            'rl_total_loss': float(total_loss.detach().cpu().item()),
            'policy_loss': float(policy_loss.detach().cpu().item()),
            'value_loss': float(value_loss.detach().cpu().item()) if isinstance(value_loss, torch.Tensor) else 0.0,
            'entropy': float(entropy.detach().mean().cpu().item()),
            'avg_reward': float(rewards_t.detach().mean().cpu().item()),
        }

        return total_loss, info

    def clear_trajectory(self):
        if self.use_rl:
            self.trajectory_buffer = {
                'states': [], 'actions': []
            }


class StandardTransformerBlock(nn.Module):
    """Standard GPT-2 block (no hypernet)"""
    
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.n_embd
        
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd, num_heads=config.n_head,
            dropout=config.attn_pdrop, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.mlp_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.mlp_act = nn.GELU()
        self.mlp_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.mlp_dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        batch_size, seq_len, _ = hidden_states.shape
        # Causal self-attention mask (shared across the batch): [S, S]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1
        )

        # Padding mask (batch-specific): [B, S], True where tokens should be ignored
        key_padding_mask = None
        if attention_mask is not None:
            # Support either a 2D (B,S) 1/0 mask or the older additive 4D mask (B,1,1,S)
            if attention_mask.dim() == 4:
                # Older format: 0 for keep, -inf (or very negative) for pad
                key_padding_mask = attention_mask[:, 0, 0, :] < 0
            else:
                # Standard format: 1 for keep, 0 for pad
                key_padding_mask = attention_mask.view(batch_size, -1) == 0
        attn_output, attn_weights = self.attn(
            hidden_states, hidden_states, hidden_states,
            attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=output_attentions
        )
        
        hidden_states = residual + attn_output
        residual = hidden_states
        
        hidden_states = self.ln_2(hidden_states)
        ffn_hidden = self.mlp_fc(hidden_states)
        ffn_hidden = self.mlp_act(ffn_hidden)
        ffn_output = self.mlp_proj(ffn_hidden)
        ffn_output = self.mlp_dropout(ffn_output)
        
        hidden_states = residual + ffn_output
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class HyperNetBlock(nn.Module):
    """Transformer block with hypernet augmentation"""
    
    def __init__(self, config: GPT2Config, layer_idx: int, hypernet: ClusteredHypernetwork):
        super().__init__()
        self.layer_idx = layer_idx
        self.hypernet = hypernet
        self.hidden_size = config.n_embd
        
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd, num_heads=config.n_head,
            dropout=config.attn_pdrop, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.mlp_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.mlp_act = nn.GELU()
        self.mlp_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.mlp_dropout = nn.Dropout(config.resid_pdrop)
        
        self.reasoning_injection = nn.Linear(hypernet.num_centroids * 3, 4 * config.n_embd, bias=False)
        self.alpha_hypernet = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, hidden_states, layer_history, attention_mask=None, 
                output_attentions=False, gate_schedule=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        batch_size, seq_len, _ = hidden_states.shape
        # Causal self-attention mask (shared across the batch): [S, S]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1
        )

        # Padding mask (batch-specific): [B, S], True where tokens should be ignored
        key_padding_mask = None
        if attention_mask is not None:
            # Support either a 2D (B,S) 1/0 mask or the older additive 4D mask (B,1,1,S)
            if attention_mask.dim() == 4:
                # Older format: 0 for keep, -inf (or very negative) for pad
                key_padding_mask = attention_mask[:, 0, 0, :] < 0
            else:
                # Standard format: 1 for keep, 0 for pad
                key_padding_mask = attention_mask.view(batch_size, -1) == 0
        attn_output, attn_weights = self.attn(
            hidden_states, hidden_states, hidden_states,
            attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=output_attentions
        )
        
        hidden_states = residual + attn_output
        residual = hidden_states
        
        hidden_states = self.ln_2(hidden_states)
        ffn_hidden = self.mlp_fc(hidden_states)
        
        # ------------------------------------------------------------------
        # HyperNet → FFN Injection (Feature Bus)
        #
        # hidden_states ──► mlp_fc (up-projection, 4*n_embd)
        #                     │
        #                     ▼
        #              ffn_hidden  ────────────────┐
        #                                          │  (+)
        # HyperNet:
        #   layer_history
        #        │
        #        ▼
        #   centroid mixture
        #        │
        #        ▼
        #   [router_out | dsin | dcos]   ∈ R^{3×128}
        #        │
        #        ▼
        #   Linear(3×128 → 4*n_embd)  ───┘  (scaled by α)
        #
        # Result:
        #   ffn_hidden := ffn_hidden + α · reasoning_injection
        #
        # Notes:
        # - Injection occurs BEFORE activation (GELU)
        # - Injects structured reasoning features into FFN subspace
        # - Preserves pretrained GPT behavior when α is small
        # ------------------------------------------------------------------

        reasoning_vector, rl_info = self.hypernet(layer_history, training=self.training)
        
        if reasoning_vector is not None:
            reasoning_injection = self.reasoning_injection(reasoning_vector)
            alpha = gate_schedule if gate_schedule is not None else self.alpha_hypernet
            ffn_hidden = ffn_hidden + alpha * reasoning_injection
        else:
            rl_info = {}
        
        ffn_hidden = self.mlp_act(ffn_hidden)
        ffn_output = self.mlp_proj(ffn_hidden)
        ffn_output = self.mlp_dropout(ffn_output)
        
        hidden_states = residual + ffn_output
        
        outputs = (hidden_states, rl_info)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class HyperNetGPT2(nn.Module):
    """
    HyperNet GPT-2 with configurable depth.

    Default config matches the original design:
      - num_layers=20
      - hypernet_start_layer=10 (layers [10..] are HyperNet-augmented)

    If you want a *pure* GPT-2-like stack with **no HyperNet blocks**, set:
      - num_layers=12
      - hypernet_start_layer=12
    (Note: this still won't be bit-identical to HuggingFace GPT-2 because the
     block implementation differs, but HyperNet is effectively disabled.)
    """

    def __init__(
        self,
        config: GPT2Config,
        num_layers: int = 12,
        hypernet_start_layer: int = 12,
        num_centroids: int = 128,
        memory_window: int = 10,
        reasoning_json_path: str = "reasoning_logic.json",
        use_rl: bool = False,
        rl_method: str = "actor_critic",
    ):
        super().__init__()

        # ---- validate ----
        num_layers = int(num_layers)
        hypernet_start_layer = int(hypernet_start_layer)
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if not (0 <= hypernet_start_layer <= num_layers):
            raise ValueError(
                f"hypernet_start_layer must be in [0, num_layers], got {hypernet_start_layer} vs {num_layers}"
            )

        self.num_layers = num_layers
        self.hypernet_start_layer = hypernet_start_layer

        config.n_layer = self.num_layers
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        self.logic_injector = None
        self.reasoning_json_path = reasoning_json_path
        self.reasoning_feature_projection = nn.Linear(4, config.n_embd)

        # HyperNet module exists even if hypernet_start_layer == num_layers.
        # In that case it's never used in forward().
        self.hypernet = ClusteredHypernetwork(
            config=config,
            num_centroids=num_centroids,
            memory_window=memory_window,
            normalize=True,
            use_rl=use_rl,
            rl_method=rl_method,
        )

        # Blocks
        self.h = nn.ModuleList()
        for i in range(self.hypernet_start_layer):
            self.h.append(StandardTransformerBlock(config, layer_idx=i))
        for i in range(self.hypernet_start_layer, self.num_layers):
            self.h.append(HyperNetBlock(config, layer_idx=i, hypernet=self.hypernet))

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        print(f"\n{'='*80}")
        print(f"✅ HyperNet GPT-2 ({self.num_layers} Layers)")
        print(f"{'='*80}")
        if self.hypernet_start_layer == self.num_layers:
            print(f"   • Layers 0-{self.num_layers-1}: Standard Transformer (HyperNet disabled)")
        elif self.hypernet_start_layer == 0:
            print(f"   • Layers 0-{self.num_layers-1}: Hypernet-Augmented")
        else:
            print(f"   • Layers 0-{self.hypernet_start_layer-1}:   Standard Transformer")
            print(f"   • Layers {self.hypernet_start_layer}-{self.num_layers-1}: Hypernet-Augmented")
        print(f"   • Centroids: {num_centroids}")
        print(f"   • RL: {'Enabled (' + rl_method + ')' if use_rl else 'Disabled'}")
        print(f"{'='*80}\n")

    def set_tokenizer(self, tokenizer):
        self.logic_injector = ReasoningLogicInjector(tokenizer, self.reasoning_json_path)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        gate_schedule=None,
    ):
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        reasoning_features = None
        if self.logic_injector is not None:
            reasoning_features = self.logic_injector.get_reasoning_features(input_ids, input_ids.device)
            reasoning_features = self.reasoning_feature_projection(reasoning_features)

        token_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = token_embeds + position_embeds

        if reasoning_features is not None:
            hidden_states = hidden_states + 0.1 * reasoning_features

        hidden_states = self.drop(hidden_states)

        # Keep attention_mask in standard (B, S) form (1 = keep, 0 = pad).
        # Padding is handled inside blocks via key_padding_mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_rl_info = []

        layer_history = []

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_history.append(hidden_states)

            if i < self.hypernet_start_layer:
                outputs = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = outputs[0]
            else:
                outputs = block(
                    hidden_states,
                    layer_history=layer_history,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    gate_schedule=gate_schedule,
                )
                hidden_states = outputs[0]
                all_rl_info.append(outputs[1])

            if output_attentions and len(outputs) > 2:
                all_attentions = all_attentions + (outputs[-1],)

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            if output_attentions:
                output = output + (all_attentions,)
            if loss is not None:
                output = (loss,) + output
            return output

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


def create_hypernet_gpt2(
    base_model: str = "gpt2",
    use_pretrained: bool = True,
    num_centroids: int = 128,
    memory_window: int = 10,
    reasoning_json_path: str = "reasoning_logic.json",
    use_rl: bool = False,
    rl_method: str = "actor_critic",
    num_layers: int | None = None,
    hypernet_start_layer: int | None = None,
):
    """Factory for HyperNetGPT2.

    Notes on defaults:
    - If num_layers / hypernet_start_layer are not provided, we preserve the
      original 20-layer (10 standard + 10 hypernet) behavior.
    - For baseline-ish GPT-2: num_layers=12, hypernet_start_layer=12.
    """

    config = GPT2Config.from_pretrained(base_model)

    # Defaults: original behavior
    if num_layers is None:
        num_layers = 12
    if hypernet_start_layer is None:
        hypernet_start_layer = 12

    config.n_layer = int(num_layers)

    model = HyperNetGPT2(
        config=config,
        num_layers=int(num_layers),
        hypernet_start_layer=int(hypernet_start_layer),
        num_centroids=num_centroids,
        memory_window=memory_window,
        reasoning_json_path=reasoning_json_path,
        use_rl=use_rl,
        rl_method=rl_method,
    )

    if use_pretrained:
        print(f"🔄 Loading pretrained weights from {base_model}...")
        pretrained = GPT2LMHeadModel.from_pretrained(base_model)

        # embeddings
        model.wte.weight.data.copy_(pretrained.transformer.wte.weight.data)
        model.wpe.weight.data.copy_(pretrained.transformer.wpe.weight.data)

        # copy blocks: up to min(num_layers, pretrained_layers)
        n_pre = int(pretrained.config.n_layer)
        n_copy = min(int(num_layers), n_pre)

        # If we have standard blocks at the front, copy those first
        n_std = min(int(hypernet_start_layer), n_copy)

        for i in range(n_std):
            pretrained_block = pretrained.transformer.h[i]
            our_block = model.h[i]

            our_block.ln_1.weight.data.copy_(pretrained_block.ln_1.weight.data)
            our_block.ln_1.bias.data.copy_(pretrained_block.ln_1.bias.data)
            our_block.ln_2.weight.data.copy_(pretrained_block.ln_2.weight.data)
            our_block.ln_2.bias.data.copy_(pretrained_block.ln_2.bias.data)

            our_block.attn.in_proj_weight.data.copy_(pretrained_block.attn.c_attn.weight.data.t())
            our_block.attn.in_proj_bias.data.copy_(pretrained_block.attn.c_attn.bias.data)
            our_block.attn.out_proj.weight.data.copy_(pretrained_block.attn.c_proj.weight.data.t())
            our_block.attn.out_proj.bias.data.copy_(pretrained_block.attn.c_proj.bias.data)

            our_block.mlp_fc.weight.data.copy_(pretrained_block.mlp.c_fc.weight.data.t())
            our_block.mlp_fc.bias.data.copy_(pretrained_block.mlp.c_fc.bias.data)
            our_block.mlp_proj.weight.data.copy_(pretrained_block.mlp.c_proj.weight.data.t())
            our_block.mlp_proj.bias.data.copy_(pretrained_block.mlp.c_proj.bias.data)

        # For remaining layers in our model, copy from matching pretrained layer
        # (or the last pretrained layer if we're deeper).
        for i in range(n_std, int(num_layers)):
            src_i = min(i, n_pre - 1)
            pretrained_block = pretrained.transformer.h[src_i]
            our_block = model.h[i]

            # blocks may be HyperNetBlock or StandardTransformerBlock, but both have these fields
            our_block.ln_1.weight.data.copy_(pretrained_block.ln_1.weight.data)
            our_block.ln_1.bias.data.copy_(pretrained_block.ln_1.bias.data)
            our_block.ln_2.weight.data.copy_(pretrained_block.ln_2.weight.data)
            our_block.ln_2.bias.data.copy_(pretrained_block.ln_2.bias.data)

            our_block.attn.in_proj_weight.data.copy_(pretrained_block.attn.c_attn.weight.data.t())
            our_block.attn.in_proj_bias.data.copy_(pretrained_block.attn.c_attn.bias.data)
            our_block.attn.out_proj.weight.data.copy_(pretrained_block.attn.c_proj.weight.data.t())
            our_block.attn.out_proj.bias.data.copy_(pretrained_block.attn.c_proj.bias.data)

            our_block.mlp_fc.weight.data.copy_(pretrained_block.mlp.c_fc.weight.data.t())
            our_block.mlp_fc.bias.data.copy_(pretrained_block.mlp.c_fc.bias.data)
            our_block.mlp_proj.weight.data.copy_(pretrained_block.mlp.c_proj.weight.data.t())
            our_block.mlp_proj.bias.data.copy_(pretrained_block.mlp.c_proj.bias.data)

        model.ln_f.weight.data.copy_(pretrained.transformer.ln_f.weight.data)
        model.ln_f.bias.data.copy_(pretrained.transformer.ln_f.bias.data)

        print("✅ Pretrained weights loaded")

    total_params = sum(p.numel() for p in model.parameters())
    hypernet_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "hypernet" in n or "centroid" in n or "reasoning_injection" in n
    )

    print(f"\n{'='*80}")
    print("📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Hypernet overhead: {hypernet_params:,} ({100*hypernet_params/total_params:.2f}%)")
    print(f"{'='*80}\n")

    return model, config
