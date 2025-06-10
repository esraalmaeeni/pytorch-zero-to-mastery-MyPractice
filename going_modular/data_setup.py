"""
Contains functionality for creating PyTorch DataLoader's for image classification data.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

num_workers = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = num_workers,
):
  """ Create training and testing DataLoaders.

  Takes train and test directory paths and turn them into pytorch datasets
  and then pytorch dataloaders.

  Args:
    train_dir: path to train dir.
    test_dir: path to test dir.
    transform: torchvision transforms to perform on train and test data.
    batch_size: no. of samples per patch in each of dataloaders.
    num_workers: an integer for no. of workers per dataloader.

    Returns:
      A tuple of (train_dataloader, test_dataloader, class_names).
      where class_names is a list of target classes.

    Example usage:
      train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir,
      test_dir=path/to/test_dir,
      transform=some_transform,
      batch_size=32,
      num_workers=1)

  """
  # Use ImageFolder to create datasets
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get the class_names
  class_names = train_data.classes

  # Turn images into DataLoaders
  train_dataloader = DataLoader(train_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True,
                                pin_memory=True)
  test_dataloader = DataLoader(test_data,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               shuffle=True,
                               pin_memory=False)

  return train_dataloader, test_dataloader, class_names
