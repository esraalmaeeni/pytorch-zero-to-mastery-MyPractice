"""File Contains various utility functions for PyTorch to train the model.
"""
from pathlib import Path
import torch
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    A function to save PyTorch model to target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A string indicating the target directory to save model to.
        model_name: A string indicating the filename for the model; should include 'pth' or 'pt'.

    Example Usage:
        save_model(model=model,
                   target_dir="models",
                   model_name="name.pth")
    """

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "model_name should end with `.pth` or `.pt`"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
