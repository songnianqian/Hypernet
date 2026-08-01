# MIT License
#
# HyperNet GPT-2: Complete Model
# 
# Includes ALL features:
# - NUM_LAYERS-layer hardcoded architecture (HYPERNET_START_LAYER hypernet)
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
        temperature: float = 5.0, 
        top_k_train: int = 1,
        top_k_eval: int = 1,
        balance_mode: str = "switch",  # switch | l2_uniform | both | off
        balance_eps: float = 1e-9,
    ):
        super().__init__()
        self.hidden_size = config.n_embd
        self.num_centroids = num_centroids
        self.memory_window = memory_window
        self.normalize = normalize
        self.use_rl = use_rl
        self.rl_method = rl_method if use_rl else None
        self.entropy_weight = entropy_weight
        self.temperature = temperature
        self.top_k_train = int(max(1, top_k_train))
        self.top_k_eval = int(max(1, top_k_eval))
        self.balance_mode = str(balance_mode or "switch").lower()
        self.balance_eps = float(balance_eps)
        
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
        
        # Centroid usage tracking
        self.register_buffer('centroid_usage_counts', torch.zeros(num_centroids, dtype=torch.long))

        # Soft-load monitoring (EMA) so we can see whether the router distribution is collapsing
        self.register_buffer('centroid_load_ema', torch.zeros(num_centroids, dtype=torch.float))
        self.load_ema_beta = 0.98
        # Expected-load style usage (float) for monitoring top-k routing; not used for control flow.
        self.register_buffer('centroid_expected_load', torch.zeros(num_centroids, dtype=torch.float))

        # ADDED: Learnable scalar to control initial Hypernet impact
        # Initializing at 0.1 ensures it doesn't overwhelm the FFN early on
        self.min_gain = 0.05          # hard coded floor (you choose)
        self.hyper_gain_raw = nn.Parameter(torch.tensor(0.0))

        # ADDED: LayerNorm to ensure M=128 distributions match Transformer stats
        # This centers and scales the router and derivative features
        self.feature_norm = nn.LayerNorm(num_centroids)
        
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
        """Compute routing distribution over centroids.

        IMPORTANT (Top-1 + Straight-Through):
        - Forward uses hard top-1 routing (one-hot over the argmax centroid).
        - Backward uses softmax gradients (straight-through estimator).
        This makes centroid training behave like k-means / VQ style responsibility assignment,
        while staying end-to-end differentiable.

        Returns:
            soft_probs:      [B,S,M] softmax(logits)  (useful for diagnostics / RL)
            routing_probs:   [B,S,M] straight-through tensor (forward=one_hot, backward=soft)
            selected:        [B,S]   argmax indices (top-1)
            selected_logp:   [B,S]   log soft prob of selected centroid
            balance_loss:    scalar  encourages uniform centroid usage (MoE-style auxiliary loss)
        """
        b, s, _ = Y_norm.shape
        centroids_norm = F.normalize(self.centroids, p=2, dim=-1)
        similarity = torch.matmul(Y_norm, centroids_norm.t())  # [b,s,M]

        if self.use_rl and training:
            Y_flat = Y_norm.reshape(b * s, self.hidden_size)
            policy_logits = self.policy_network(Y_flat).reshape(b, s, self.num_centroids)
            logits = similarity + policy_logits
        else:
            logits = similarity

        # --- TEMPERATURE SCALING ---
        # T > 1.0 flattens the distribution (helps dead centroids)
        # T < 1.0 sharpens it (focuses on winners)
        temperature = self.temperature if training else 1.0 
        logits = logits / temperature

        # Soft distribution (used for balance loss + optional straight-through gradients)
        soft_probs = F.softmax(logits, dim=-1)  # [B,S,M]

        # ------------------------------------------------------------
        # Top-k routing (k=1 by default).
        # - forward uses a *sparse* distribution over the top-k centroids
        # - backward uses full-softmax gradients (straight-through)
        #
        # This is a practical compromise:
        #   * k=1 behaves like VQ / k-means responsibility assignment.
        #   * k=2 gives exploration + smoother usage without full dense MoE.
        # ------------------------------------------------------------
        k = self.top_k_train if training else self.top_k_eval
        k = int(max(1, min(k, self.num_centroids)))

        if k == 1:
            selected = torch.argmax(soft_probs, dim=-1)  # [B,S]
            sparse = F.one_hot(selected, num_classes=self.num_centroids).to(dtype=soft_probs.dtype)  # [B,S,M]
        else:
            topk_vals, topk_idx = torch.topk(soft_probs, k=k, dim=-1)  # [B,S,k]
            # Normalize top-k weights so forward still sums to 1.0
            topk_w = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + self.balance_eps)
            sparse = torch.zeros_like(soft_probs)
            sparse.scatter_(dim=-1, index=topk_idx, src=topk_w)
            # For reporting / usage_counts we still define a single 'winner'
            selected = topk_idx[..., 0]

        # Straight-through: forward=sparse, backward=soft_probs gradients
        routing_probs = sparse + soft_probs - soft_probs.detach()

        # log prob of selected action (always based on soft_probs)
        log_probs = torch.log(soft_probs + 1e-10)
        selected_logp = log_probs.gather(dim=2, index=selected.unsqueeze(-1)).squeeze(-1)  # [B,S]

        # ------------------------------------------------------------
        # MoE-style load balancing loss
        #
        # Recommended (Switch Transformer-style):
        #   importance = mean soft_probs
        #   load       = mean (forward routing)  (top-1 one_hot or top-k sparse)
        #   loss = M * sum(importance * load)
        #
        # This penalizes collapse where a few experts are both highly probable
        # and heavily used.
        # ------------------------------------------------------------
        mode = (self.balance_mode or "switch").lower()
        if mode in ("off", "none", "0"):
            balance_loss = logits.new_zeros(())
        else:
            importance = soft_probs.mean(dim=(0, 1))  # [M]
            load = sparse.detach().mean(dim=(0, 1))   # [M]

            # Switch-style term
            switch_loss = float(self.num_centroids) * torch.sum(importance * load)

            # Optional L2-to-uniform regularizer on load (your previous behavior)
            target = 1.0 / float(self.num_centroids)
            l2_uniform = ((load - target) ** 2).mean()

            if mode == "switch":
                balance_loss = switch_loss
            elif mode == "l2_uniform":
                balance_loss = l2_uniform
            elif mode == "both":
                balance_loss = switch_loss + l2_uniform
            else:
                # default to switch-style
                balance_loss = switch_loss

        return soft_probs, routing_probs, selected, selected_logp, balance_loss

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

        # Numerical stability (especially under AMP):
        # shift logits by per-token max before softmax to avoid exp overflow.
        logits = logits - logits.max(dim=-1, keepdim=True).values

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
        
        # Clamp to prevent extreme values
        Y_aggregated = torch.clamp(Y_aggregated, -10, 10)
        sX_aggregated = torch.clamp(sX_aggregated, -10, 10)
        cX_aggregated = torch.clamp(cX_aggregated, -10, 10)
        
        Y_norm = F.normalize(Y_aggregated, p=2, dim=-1, eps=1e-8)
        
        action_probs, routing_probs, selected_centroids, log_probs, balance_loss = self.get_centroid_distribution(
            Y_norm, training=training
        )
        
        # Track centroid usage
        if training:
            # Count how many times each centroid is selected
            with torch.no_grad():
                counts = torch.bincount(
                    selected_centroids.reshape(-1),
                    minlength=self.num_centroids
                ).to(self.centroid_usage_counts.device)
                self.centroid_usage_counts += counts
        
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
        centroid_weights_flat = routing_probs.reshape(b * s, self.num_centroids)
        router_out = self.router_perceptron(centroid_weights_flat).reshape(b, s, self.num_centroids)
        
        # Derivative features (computed from aggregated sin/cos components)
        dsin = cX_aggregated
        dcos = -sX_aggregated
        dsin_norm = F.normalize(dsin, p=2, dim=-1, eps=1e-8)
        dcos_norm = F.normalize(dcos, p=2, dim=-1, eps=1e-8)
        centroids_norm = F.normalize(self.centroids, p=2, dim=-1, eps=1e-8)
        sim_dsin = torch.matmul(dsin_norm, centroids_norm.t())
        sim_dcos = torch.matmul(dcos_norm, centroids_norm.t())
        # Clamp similarities to prevent extreme softmax values
        sim_dsin = torch.clamp(sim_dsin, -10, 10)
        sim_dcos = torch.clamp(sim_dcos, -10, 10)
        dsin_probs = F.softmax(sim_dsin, dim=-1)
        dcos_probs = F.softmax(sim_dcos, dim=-1)
        
        # NEW: Apply Normalization and Scaling
        # Normalize each 128-d component so they are N(0,1) like GPT-2 activations
        router_out = self.feature_norm(router_out)
        dsin_probs = self.feature_norm(dsin_probs)
        dcos_probs = self.feature_norm(dcos_probs)
        
        # Final routing features [B, S, 384]
        reasoning_vector = torch.cat([router_out, dsin_probs, dcos_probs], dim=-1)
        if getattr(self, "disable_hypernet", False):
            reasoning_vector = None
        
        # Apply the learnable gain (starts at 0.1)
        if reasoning_vector is not None:
            gain = self.min_gain + F.softplus(self.hyper_gain_raw)
            reasoning_vector = reasoning_vector * gain

        rl_info = {
            'action_probs': action_probs,
            'routing_probs': routing_probs,
            'selected_centroids': selected_centroids,
            'log_probs': log_probs,
            'values': values,
            'entropy': entropy,
            'states': Y_norm,
            'balance_loss': balance_loss,
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

            # -----------------------------
            # NaN/Inf guardrails (stable RL)
            # -----------------------------
            # selected_log_probs are log π(a|s). Extremely large magnitude values
            # can destabilize AMP and cause NaN/Inf gradients. Clamp them.
            selected_log_probs = torch.clamp(selected_log_probs, -30.0, 30.0)
            # Rewards/advantages can spike (especially early); clamp to keep
            # policy gradients bounded.
            advantages = torch.clamp(advantages, -5.0, 5.0)

            policy_loss = -(selected_log_probs * advantages.detach()).mean()
            value_loss = torch.tensor(0.0, device=policy_loss.device)
        else:
            # actor_critic (default)
            if values is None:
                raise RuntimeError("actor_critic selected but value_network is missing")
            advantages = rewards_t - values

            # -----------------------------
            # NaN/Inf guardrails (stable RL)
            # -----------------------------
            selected_log_probs = torch.clamp(selected_log_probs, -30.0, 30.0)
            advantages = torch.clamp(advantages, -5.0, 5.0)

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
    
    def get_centroid_usage(self, topk: int = 10, reset: bool = False):
        """
        Get centroid usage statistics.
        
        Args:
            topk: Number of top centroids to return
            reset: Whether to reset counts after retrieval
        
        Returns:
            Dictionary with 'total' counts and 'top' centroid indices
        """
        usage_dict = {
            'total': self.centroid_usage_counts.cpu().tolist(),
            'top': torch.topk(self.centroid_usage_counts, min(topk, self.num_centroids)).indices.cpu().tolist()
        }
        
        if reset:
            self.centroid_usage_counts.zero_()
        
        return usage_dict


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

        # --- causal mask: [S, S], bool ---
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1
        )

        # --- key padding mask: [B, S], bool ---
        key_padding_mask = None
        if attention_mask is not None:
            # Ensure attention_mask is 2-D: [B, S]
            if attention_mask.dim() > 2:
                # If 4-D [B, 1, S, S] or 3-D [B, 1, S], squeeze extra dims
                attention_mask = attention_mask.squeeze(1)
                if attention_mask.dim() > 2:
                    # If still 3-D [B, S, S], take diagonal or first row
                    attention_mask = attention_mask[:, 0, :]
            key_padding_mask = ~attention_mask.bool()

        attn_output, attn_weights = self.attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=causal_mask,              # [S, S]
            key_padding_mask=key_padding_mask,  # [B, S]
            need_weights=output_attentions,
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
        
        # MODIFIED: Extended projection to accommodate concatenated hypernet features
        # Original: [768, 3072]
        # New: [768, 3072 + 384] = [768, 3456]
        hypernet_feature_dim = hypernet.num_centroids * 3  # 384
        extended_dim = 4 * config.n_embd + hypernet_feature_dim  # 3456
        
        self.mlp_proj_extended = nn.Linear(extended_dim, config.n_embd)
        self.mlp_dropout = nn.Dropout(config.resid_pdrop)
        
        # Will copy pretrained weights to first 3072 columns in create_hypernet_gpt2()
        with torch.no_grad():
            nn.init.normal_(self.mlp_proj_extended.weight[:, 4*config.n_embd:], mean=0.0, std=1e-3)
        
        print(f"  Layer {layer_idx}: Using concatenation approach (FFN dim: {extended_dim})")
    
    def forward(self, hidden_states, attention_mask=None, 
                output_attentions=False):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Create causal mask [S, S]
        # --- causal mask: [S, S], bool ---
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1
        )

        # For MultiheadAttention with batch_first=True:
        # - attn_mask: 2-D [S, S] for causal masking (same for all batches)
        # - key_padding_mask: 2-D [B, S] for padding (True = ignore position)
        key_padding_mask = None
        if attention_mask is not None and attention_mask.dim() == 2:
            # [B, S] -> invert for key_padding_mask (True = ignore)
            key_padding_mask = ~attention_mask.bool()
        
        attn_output, attn_weights = self.attn(
            hidden_states, hidden_states, hidden_states,
            attn_mask=causal_mask,  # 2-D [S, S]
            key_padding_mask=key_padding_mask,  # 2-D [B, S] or None
            need_weights=output_attentions
        )
        
        hidden_states = residual + attn_output
        residual = hidden_states
        
        hidden_states = self.ln_2(hidden_states)
        
        # ------------------------------------------------------------------
        # MODIFIED: Concatenation approach instead of addition
        # 
        # Standard FFN path:
        #   hidden_states → mlp_fc → ffn_hidden [B, S, 3072]
        #                                         ↓
        #                                       GELU
        #                                         ↓
        # HyperNet path:
        #   layer_history → hypernet → reasoning_vector [B, S, 384]
        # 
        # Concatenation:
        #   [ffn_hidden | reasoning_vector] → [B, S, 3456]
        #                                         ↓
        #   mlp_proj_extended [768, 3456] → output [B, S, 768]
        # 
        # Benefits:
        #   - Hypernet has independent channels (no interference)
        #   - Network learns how much to use hypernet (via weights)
        #   - If hypernet weights ≈ 0, model = baseline GPT-2
        #   - More stable (NaN in hypernet doesn't break base FFN)
        # ------------------------------------------------------------------
        
        ffn_hidden = self.mlp_fc(hidden_states)  # [B, S, 3072]
        ffn_hidden = self.mlp_act(ffn_hidden)
        
        # Get hypernet features
        # NOTE (No-history routing): feed ONLY the current layer hidden_states into the shared HyperNet.
        # This removes layer_history averaging/statefulness while keeping all other HyperNet features intact.
        reasoning_vector, rl_info = self.hypernet([hidden_states], training=self.training)
        
        if reasoning_vector is not None:
            # Concatenate base FFN and hypernet features
            ffn_expanded = torch.cat([ffn_hidden, reasoning_vector], dim=-1)  # [B, S, 3456]
        else:
            # If no hypernet output, pad with zeros
            zero_padding = torch.zeros(
                batch_size, seq_len, self.hypernet.num_centroids * 3,
                device=ffn_hidden.device, dtype=ffn_hidden.dtype
            )
            ffn_expanded = torch.cat([ffn_hidden, zero_padding], dim=-1)
            rl_info = {}
        
        # Project back to embedding dimension
        # The last 384 columns of mlp_proj_extended control hypernet contribution
        ffn_output = self.mlp_proj_extended(ffn_expanded)  # [B, S, 768]
        ffn_output = self.mlp_dropout(ffn_output)
        
        hidden_states = residual + ffn_output
        
        outputs = (hidden_states, rl_info)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class HyperNetGPT2(nn.Module):
    """
    NUM_LAYERS-Layer HyperNet GPT-2 with all features.
    
    Architecture:
    - Layers NUM_LAYERS-HYPERNET_START_LAYER: Standard transformer (context building)
    - Layers HYPERNET_START_LAYER-NUM_LAYERS: Hypernet-augmented (reasoning)
    - Optional RL for centroid selection
    - Logic injection
    - All architectural improvements
    """
    
    NUM_LAYERS = 12
    HYPERNET_START_LAYER = 6
    
    def __init__(
        self,
        config: GPT2Config,
        num_centroids: int = 128,
        memory_window: int = 10,
        reasoning_json_path: str = "reasoning_logic.json",
        use_rl: bool = False,
        rl_method: str = 'actor_critic',
        balance_weight: float = 0.01,
        temperature: float = 5.0, 
        top_k_train: int = 1,
        top_k_eval: int = 1,
        balance_mode: str = "switch",
    ):
        super().__init__()

        print(
            f"HyperNetGPT2 config: "
            f"NUM_LAYERS={self.NUM_LAYERS}, "
            f"HYPERNET_START_LAYER={self.HYPERNET_START_LAYER}"
        )
        
        config.n_layer = self.NUM_LAYERS
        self.config = config
        self.balance_weight = float(balance_weight)
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        self.logic_injector = None
        self.reasoning_json_path = reasoning_json_path
        self.reasoning_feature_projection = nn.Linear(4, config.n_embd)
        
        self.hypernet = ClusteredHypernetwork(
            config=config,
            num_centroids=num_centroids,
            memory_window=memory_window,
            normalize=True,
            use_rl=use_rl,
            rl_method=rl_method,
            temperature=temperature, 
            top_k_train=top_k_train,
            top_k_eval=top_k_eval,
            balance_mode=balance_mode,
        )
        
        self.h = nn.ModuleList()
        for i in range(self.HYPERNET_START_LAYER):
            self.h.append(StandardTransformerBlock(config, layer_idx=i))
        for i in range(self.HYPERNET_START_LAYER, self.NUM_LAYERS):
            self.h.append(HyperNetBlock(config, layer_idx=i, hypernet=self.hypernet))
        
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        
        print(f"\n{'='*80}")
        print(f"✅ HyperNet GPT-2 (NUM_LAYERS Layers)")
        print(f"{'='*80}")
        print(f"   • Layers 0-6:   Standard Transformer")
        print(f"   • Layers 7-12: Hypernet-Augmented")
        print(f"   • Centroids: {num_centroids}")
        print(f"   • RL: {'Enabled (' + rl_method + ')' if use_rl else 'Disabled'}")
        print(f"{'='*80}\n")
    
    def set_tokenizer(self, tokenizer):
        self.logic_injector = ReasoningLogicInjector(tokenizer, self.reasoning_json_path)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None,
                output_attentions=False, output_hidden_states=False, 
                return_dict=True):
        
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        reasoning_features = None
        if self.logic_injector is not None:
            reasoning_features = self.logic_injector.get_reasoning_features(
                input_ids, input_ids.device
            )
            reasoning_features = self.reasoning_feature_projection(reasoning_features)
        
        token_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = token_embeds + position_embeds
        
        if reasoning_features is not None:
            hidden_states = hidden_states + 0.1 * reasoning_features
        
        hidden_states = self.drop(hidden_states)
        
        # Don't modify attention_mask here - it's already handled properly in blocks
        # The blocks expect 2-D attention_mask [B, S] for key_padding_mask
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_rl_info = []
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if i < self.HYPERNET_START_LAYER:
                outputs = block(hidden_states, attention_mask=attention_mask,
                              output_attentions=output_attentions)
                hidden_states = outputs[0]
            else:
                outputs = block(hidden_states,
                attention_mask=attention_mask,
                              output_attentions=output_attentions)
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
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1))


        # ------------------------------------------------------------------
        # MoE-style load balancing loss (encourage uniform centroid usage)
        # We aggregate balance_loss from each HyperNetBlock call.
        # This is lightweight and helps prevent centroid collapse / dead centroids.
        # ------------------------------------------------------------------
        if loss is not None and len(all_rl_info) > 0 and self.balance_weight > 0:
            bal_terms = []
            for info in all_rl_info:
                if isinstance(info, dict) and ('balance_loss' in info) and (info['balance_loss'] is not None):
                    bal_terms.append(info['balance_loss'])
            if len(bal_terms) > 0:
                total_balance_loss = torch.stack(bal_terms).mean()
                loss = loss + self.balance_weight * total_balance_loss
        
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
    rl_method: str = 'actor_critic',
    temperature: float = 5.0,  #
    top_k_train: int = 1,
    top_k_eval: int = 1,
    balance_weight: float = 0.01,
    balance_mode: str = "switch",
):
    """
    Create HyperNet GPT-2 with all features.
    
    Args:
        base_model: Base model for config
        use_pretrained: Load pretrained weights
        num_centroids: Number of centroids (M=128)
        memory_window: Memory window for hypernet
        reasoning_json_path: Path to reasoning logic JSON
        use_rl: Enable RL for centroid selection
        rl_method: 'reinforce', 'actor_critic', or 'advantage'
    
    Returns:
        model: HyperNet GPT-2
        config: Model configuration
    """
    
    config = GPT2Config.from_pretrained(base_model)
    
    model = HyperNetGPT2(
        config=config,
        num_centroids=num_centroids,
        memory_window=memory_window,
        reasoning_json_path=reasoning_json_path,
        use_rl=use_rl,
        rl_method=rl_method,
        balance_weight=balance_weight,
        temperature=temperature,
        top_k_train=top_k_train,
        top_k_eval=top_k_eval,
        balance_mode=balance_mode,
    )
    config.n_layer = model.NUM_LAYERS

    if use_pretrained:
        print(f"🔄 Loading pretrained weights from {base_model}...")
        pretrained = GPT2LMHeadModel.from_pretrained(base_model)
        
        model.wte.weight.data.copy_(pretrained.transformer.wte.weight.data)
        model.wpe.weight.data.copy_(pretrained.transformer.wpe.weight.data)
        
        for i in range(min(model.HYPERNET_START_LAYER, pretrained.config.n_layer)):
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
        
        for i in range(model.HYPERNET_START_LAYER, model.NUM_LAYERS):
            pretrained_idx = min(10 + (i - 10) % 2, pretrained.config.n_layer - 1)
            pretrained_block = pretrained.transformer.h[pretrained_idx]
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
            
            # MODIFIED: Copy pretrained weights to first 3072 columns of extended projection
            # mlp_proj_extended is [768, 3456], pretrained is [768, 3072]
            # Copy to columns 0:3072 (base FFN), leave columns 3072:3456 (hypernet) as zero
            our_block.mlp_proj_extended.weight.data[:, :4*config.n_embd].copy_(
                pretrained_block.mlp.c_proj.weight.data.t()
            )
            our_block.mlp_proj_extended.bias.data.copy_(pretrained_block.mlp.c_proj.bias.data)
            # Columns 3072:3456 already initialized to zero in __init__
        
        model.ln_f.weight.data.copy_(pretrained.transformer.ln_f.weight.data)
        model.ln_f.bias.data.copy_(pretrained.transformer.ln_f.bias.data)
        
        print("✅ Pretrained weights loaded")
    
    total_params = sum(p.numel() for p in model.parameters())
    hypernet_params = sum(p.numel() for n, p in model.named_parameters() 
                         if 'hypernet' in n or 'centroid' in n or 'reasoning_injection' in n)
    
    print(f"\n{'='*80}")
    print(f"📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Hypernet overhead: {hypernet_params:,} ({100*hypernet_params/total_params:.2f}%)")
    print(f"{'='*80}\n")
    
    return model, config


if __name__ == "__main__":
    print("Creating HyperNet GPT-2...")
    
    model, config = create_hypernet_gpt2(
        base_model="gpt2",
        use_pretrained=False,
        use_rl=False,
    )
    
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.set_tokenizer(tokenizer)
    
    test_text = "If x + 2 = 5 then x = 3"
    input_ids = tokenizer.encode(test_text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    
    print(f"✅ Test successful! Loss: {outputs.loss.item():.4f}")