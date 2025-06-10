"""
Contains functions for training and testing PyTorch.
"""

import torch
from typing import Tuple, List, Dict
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then runs through all
    required training steps (forward pass, loss calculation, optimizer step.)

    Args:
        model: A PyTorch model to be trained on.
        dataloader: A dataloader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu")

    Returns:
        A tuple of training loss and training accuracy metrics, in
        the form (train_loss, train_acc) e.g. (0.3335, 0.8456)

    Example usage:
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
    """
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate and accumulate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to eval mode and then performs
    a forward pass on the testing dataset and calculates loss and accuracy.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A dataloader instance for the test dataset.
        loss_fn: A PyTorch loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu")

    Returns:
        A tuple of testing loss and testing accuracy metrics, in
        the form (test_loss, test_acc) e.g. (0.3335, 0.8456)

    Example usage:
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
    """
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(X)

            # Loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Accuracy
            test_pred_labels = torch.argmax(test_pred_logits, dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
    """Runs training and testing for a number of epochs and tracks metrics.

    Args:
        model: A PyTorch model to be tested.
        train_dataloader: A train dataloader instance for the train dataset.
        test_dataloader: A test dataloader instance for the test dataset.
        loss_fn: A PyTorch loss function to minimiz.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        epochs: An integer indicating the number of epochs to be trained for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A Dictionary of training and testing loss as well as training and testing accuracy metrics,
        each metric has values in the form of list, in the form:
            {
              "train_loss": [...],
              "train_acc": [...],
              "test_loss": [...],
              "test_acc": [...]
            }

        Example, if number of epochs is 3:
            {
              "train_loss": [0.4352, 0.3456, 0.2357],
              "train_acc": [0.8765, 0.8854, 0.8976],
              "test_loss": [0.7658, 0.6789, 0.5498],
              "test_acc": [0.7895, 0.7890, 0.7998]
            }


    Example usage:
        Results = train(
            model=model,
            train_dataloader= path/to/train/DataLoader,
            test_dataloader= path/to/test/DataLoader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs= 5,
            device=device
        )
    """

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
