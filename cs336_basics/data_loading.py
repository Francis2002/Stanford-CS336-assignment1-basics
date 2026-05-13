import torch
from torch.utils.data import Dataset

class TokenDataset(Dataset):
    """
    A Dataset that samples random spans from a tokenized array.
    This is designed to work with DataLoader for parallel pre-fetching.
    """
    def __init__(self, token_array, context_length):
        self.x = token_array
        self.context_length = context_length

    def __len__(self):
        return len(self.x) - self.context_length

    def __getitem__(self, idx):
        # Discard the dataloader-provided index to sample completely randomly 
        # from the memmapped array. This acts as an infinite stream of random spans.
        start_idx = torch.randint(0, len(self.x) - self.context_length, (1,)).item()
        
        # Slicing numpy memmap is fast
        inputs = torch.from_numpy(self.x[start_idx : start_idx + self.context_length].astype("int64"))
        targets = torch.from_numpy(self.x[start_idx + 1 : start_idx + self.context_length + 1].astype("int64"))
        
        return inputs, targets

def load_data(x, batch_size, context_length, device):
    """
    Legacy helper for synchronous sampling. 
    Maintains compatibility for simple scripts.
    """
    starting_indices = torch.randint(0, len(x) - context_length, (batch_size,))
    inputs = torch.stack([torch.from_numpy(x[idx : idx + context_length].astype("int64")) for idx in starting_indices])
    targets = torch.stack([torch.from_numpy(x[idx + 1 : idx + context_length + 1].astype("int64")) for idx in starting_indices])
    return inputs.to(device), targets.to(device)