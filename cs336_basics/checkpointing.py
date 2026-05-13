import torch
import os
import json
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, iteration, out, config=None):
    """
    Safely saves the model and optimizer states via atomic write.
    
    Args:
        model (nn.Module): The model to save.
        optimizer (optim.Optimizer): The optimizer state.
        iteration (int): Current training step.
        out (str | PathLike): Destination file path.
        config (dict, optional): Run configuration to save alongside the checkpoint.
    """
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    temp_path = str(out) + ".tmp"
    
    checkpoint = {
        'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
        'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else optimizer,
        'iteration': iteration
    }
    
    # Atomic write
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, out)
    
    # Save config alongside the first checkpoint in a run directory if provided
    if config is not None:
        config_path = os.path.join(out_dir, "config.json")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved run configuration to {config_path}")

def load_checkpoint(src, model, optimizer=None, device=None, prefix="_orig_mod."):
    """
    should load a checkpoint from src (path or file-
    like object), and then recover the model and optimizer states from that checkpoint. Your
    function should return the iteration number that was saved to the checkpoint. You can use
    torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
    load_state_dict method in both the model and optimizers to return them to their previous
    states.
    This function expects the following parameters:
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
        device: torch.device
    prefix: str
    """
    checkpoint = torch.load(src, map_location=device)

    # Strip "_orig_mod." prefix added by torch.compile() if present
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']