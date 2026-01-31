import torch
import numpy as np

def load_data(x, batch_size, context_length, device):
    """
    Write a function that takes a numpy array x (integer array with token IDs), a
    batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returns
    a pair of tensors: the sampled input sequences and the corresponding next-token targets. Both ten-
    sors should have shape (batch_size, context_length) containing token IDs, and both should be
    placed on the requested device.
    """
    
    # Randomly select starting indices for the batch_size sequences
    starting_indices = torch.randint(0, len(x) - context_length, (batch_size,))

    # Initialize input sequences and targets
    inputs = np.zeros((batch_size, context_length), dtype=np.int64)
    targets = np.zeros((batch_size, context_length), dtype=np.int64)

    # Fill in the input sequences and targets
    for i, start_idx in enumerate(starting_indices):
        inputs[i] = x[start_idx : start_idx + context_length]
        targets[i] = x[start_idx + 1 : start_idx + context_length + 1]

    # Pass convert from numpy to tensors and place on device
    inputs = torch.from_numpy(inputs).to(device)
    targets = torch.from_numpy(targets).to(device)

    return inputs, targets