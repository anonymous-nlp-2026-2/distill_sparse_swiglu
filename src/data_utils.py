import os
# Data loading: WikiText-103 (local cache) for training, WikiText-2/C4 for evaluation.
# All loaders tokenize and pack text into fixed-length sequences.

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset


class PackedTokenDataset(IterableDataset):
    """Streams text, tokenizes, and packs into fixed-length (seq_len) chunks."""

    def __init__(self, hf_dataset, tokenizer, seq_len: int, seed: int = 42):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            text = example.get("text", "")
            if not text.strip():
                continue
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def get_calibration_loader(tokenizer, batch_size: int, seq_len: int, seed: int = 42,
                          dataset_name: str = "wikitext-103"):
    if dataset_name == "wikitext-103":
        _wt103_local = "/root/autodl-tmp/data/wikitext103/"
        if os.path.isdir(_wt103_local):
            from datasets import load_from_disk
            ds = load_from_disk(_wt103_local)
        else:
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        ds = ds.shuffle(seed=seed)
    elif dataset_name == "c4":
        _c4_local = "/root/autodl-tmp/data/c4_val"
        if os.path.isdir(_c4_local):
            from datasets import load_from_disk
            ds = load_from_disk(_c4_local)
            ds = ds.shuffle(seed=seed)
        else:
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=10000)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    packed = PackedTokenDataset(ds, tokenizer, seq_len, seed)
    return DataLoader(packed, batch_size=batch_size, num_workers=0)



class RandomTokenDataset(IterableDataset):
    """Random token data for dry-run testing."""

    def __init__(self, vocab_size: int, seq_len: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.rng = torch.Generator().manual_seed(seed)

    def __iter__(self):
        while True:
            tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=self.rng)
            yield {"input_ids": tokens[:-1], "labels": tokens[1:]}


def get_random_loader(vocab_size: int, batch_size: int, seq_len: int, seed: int = 42):
    """Random DataLoader for dry-run testing without network access."""
    ds = RandomTokenDataset(vocab_size, seq_len, seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


def get_eval_dataset(name: str, tokenizer, seq_len: int, max_samples: int = None):
    """Load and pack an evaluation dataset.

    Args:
        name: "wikitext2" or "c4"
        tokenizer: HF tokenizer
        seq_len: target sequence length
        max_samples: cap on number of packed sequences returned

    Returns:
        list of dicts with "input_ids" and "labels" tensors (each shape [seq_len])
    """
    if name == "wikitext2":
        # Local fallback for offline environments (HF Hub unreachable)
        _wt2_local = "/root/autodl-tmp/data/wikitext2/test"
        if os.path.isdir(_wt2_local):
            from datasets import load_from_disk
            ds = load_from_disk(_wt2_local)
        else:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [ex["text"] for ex in ds if ex["text"].strip()]
    elif name == "c4":
        _c4_local = "/root/autodl-tmp/data/c4_val"
        if os.path.isdir(_c4_local):
            from datasets import load_from_disk
            ds = load_from_disk(_c4_local)
        else:
            ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        limit = (max_samples or 200) * 3
        for i, ex in enumerate(ds):
            if i >= limit:
                break
            if ex["text"].strip():
                texts.append(ex["text"])
    else:
        raise ValueError(f"Unknown dataset: {name}")

    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer(text, add_special_tokens=False)["input_ids"])

    examples = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        chunk = all_tokens[i : i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        examples.append(
            {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
            }
        )
        if max_samples and len(examples) >= max_samples:
            break

    return examples
