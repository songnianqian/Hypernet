# dyck_dataset.py
import random
from torch.utils.data import Dataset
import torch

class DyckDataset(Dataset):
    """
    LM-only Dyck-k dataset (balanced brackets).
    Produces:
      input_ids: [max_length]
      attention_mask: [max_length] (bool)
    """
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        n_samples: int = 100000,
        max_depth: int = 20,
        p_continue: float = 0.7,
        dyck_types: int = 2,
        # Stop control:
        # - stop_prob: probability of stopping when stack becomes empty (after meeting min_depth)
        # - min_depth: require reaching at least this max stack depth before allowing early stop
        stop_prob: float = 0.2,
        min_depth: int = 0,
        seed: int = 1234,
    ):
        self.tok = tokenizer
        self.max_length = max_length
        self.n_samples = n_samples
        self.max_depth = max_depth
        self.p_continue = p_continue
        self.dyck_types = dyck_types
        self.stop_prob = float(stop_prob)
        self.min_depth = int(min_depth)
        self.rng = random.Random(seed)

        if dyck_types == 1:
            self.opens = ["("]
            self.closes = [")"]
        elif dyck_types == 2:
            self.opens = ["(", "["]
            self.closes = [")", "]"]
        elif dyck_types == 4:
            self.opens = ["(", "[", "{", "<"]
            self.closes = [")", "]", "}", ">"]
        else:
            raise ValueError("dyck_types must be 1, 2, or 4")
    
        # Ensure pad token exists
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def __len__(self):
        return self.n_samples

    def _gen_balanced_symbols(self):
        stack = []
        out = []
        maxd_seen = 0

        # keep generating until we decide to stop and stack is empty
        while True:
            can_open = len(stack) < self.max_depth
            can_close = len(stack) > 0

            # decide action
            if can_open and (not can_close or self.rng.random() < self.p_continue):
                t = self.rng.randrange(self.dyck_types)
                stack.append(t)
                if len(stack) > maxd_seen:
                    maxd_seen = len(stack)
                out.append(self.opens[t])
            else:
                t = stack.pop()
                out.append(self.closes[t])

            # stop condition: empty stack and random stop (only after reaching min_depth),
            # or too long
            if len(stack) == 0:
                # Only allow early stop if we've reached the required depth at least once.
                if maxd_seen >= self.min_depth and self.rng.random() < self.stop_prob:
                    break
                # Otherwise, keep going (this forces deeper sequences during eval when min_depth > 0).
            if len(out) >= (self.max_length * 2):  # safety; we’ll truncate later
                # close everything
                while stack and len(out) < (self.max_length * 2):
                    t = stack.pop()
                    out.append(self.closes[t])
                break

        return out

    def __getitem__(self, idx):
        syms = self._gen_balanced_symbols()

        # Space-separated to stabilize GPT-2 BPE
        text = " ".join(syms)

        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].squeeze(0)            # [L]
        attn = enc["attention_mask"].squeeze(0).to(torch.bool)

        # Pad to max_length
        if input_ids.numel() < self.max_length:
            pad_len = self.max_length - input_ids.numel()
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tok.pad_token_id, dtype=torch.long)])
            attn = torch.cat([attn, torch.zeros(pad_len, dtype=torch.bool)])

        return {"input_ids": input_ids, "attention_mask": attn}

    def dyck_max_depth(symbols, opens, closes):
        open_set = set(opens)
        close_map = {closes[i]: opens[i] for i in range(len(opens))}
        stack = []
        maxd = 0
        for s in symbols:
            if s in open_set:
                stack.append(s)
                if len(stack) > maxd:
                    maxd = len(stack)
            elif s in close_map:
                if stack and stack[-1] == close_map[s]:
                    stack.pop()
        return maxd
