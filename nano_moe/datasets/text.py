import torch
from torch.utils.data import DataLoader, Dataset
import os

try:
    from transformers import AutoTokenizer
    from datasets import load_dataset as hf_load_dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers or Datasets library not found. Text training will not work.")

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        # Each chunk needs seq_len + 1 tokens (inputs plus the shifted target),
        # so the final partial chunk is dropped.
        return max(0, (len(self.tokens) - 1) // self.seq_len)
    
    def __getitem__(self, idx):
        # Non-overlapping chunks; y is x shifted by one token.
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class CharTokenizer:
    """Character-level tokenizer built from the characters of a text."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str):
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def get_char_loaders(path: str, batch_size: int, seq_len: int, num_workers=0,
                     val_frac: float = 0.1, max_chars=None, seed: int = 0):
    """Build character-level train/val loaders from a local text file.

    Works offline (no tokenizer download), which makes it the default for
    CPU smoke runs on ``input.txt``. The split is contiguous (last
    ``val_frac`` of the text is validation) so no validation text leaks into
    training chunks.

    Returns:
        (loaders, vocab_size, tokenizer) where ``loaders`` has the same
        ``{name: {"train_loader", "test_loader"}}`` layout as
        :func:`get_text_loaders`.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if max_chars is not None:
        text = text[:max_chars]
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(len(data) * val_frac)
    train_tokens, val_tokens = data[:-n_val], data[-n_val:]

    g = torch.Generator().manual_seed(seed)
    loaders = {
        path: {
            "train_loader": DataLoader(TextDataset(train_tokens, seq_len), batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers, generator=g),
            "test_loader": DataLoader(TextDataset(val_tokens, seq_len), batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers),
        }
    }
    return loaders, tokenizer.vocab_size, tokenizer


def get_text_loaders(dataset_name: str, batch_size: int, seq_len: int, num_workers=0):
    """GPT-2 BPE loaders for Hugging Face text datasets (needs network access)."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library required for text loaders.")
        
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if dataset_name.lower() in ["wikitext", "wikitext-103"]:
        dataset = hf_load_dataset("wikitext", "wikitext-103-raw-v1")
        text_col = "text"
    elif dataset_name.lower() in ["shakespeare", "tinyshakespeare", "tiny_shakespeare"]:
        try:
            dataset = hf_load_dataset("tiny_shakespeare")
        except Exception:
            print("Could not load tiny_shakespeare from HF, falling back to wikitext-2")
            dataset = hf_load_dataset("wikitext", "wikitext-2-raw-v1")
        text_col = "text"
    else:
        # Default to wikitext-2
        dataset = hf_load_dataset("wikitext", "wikitext-2-raw-v1")
        text_col = "text"
        
    def tokenize_function(examples):
        return tokenizer(examples[text_col])
    
    # Tokenize
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=[text_col])
    
    # Concatenate all tokens
    # This is a simplified approach: just concat everything and chunk
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        return concatenated_examples

    # We'll just take the train split and flatten it for simplicity in this demo
    train_data = tokenized_datasets["train"]
    val_data = tokenized_datasets["validation"]
    
    # Flatten (inefficient for huge datasets but fine for demo)
    train_tokens = torch.tensor([item for sublist in train_data['input_ids'] for item in sublist], dtype=torch.long)
    val_tokens = torch.tensor([item for sublist in val_data['input_ids'] for item in sublist], dtype=torch.long)
    
    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)
    
    loaders = {
        dataset_name: {
            "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            "test_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }
    }
    
    return loaders, tokenizer.vocab_size
