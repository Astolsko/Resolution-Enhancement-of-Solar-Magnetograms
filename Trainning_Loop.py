# Preferred Optimizer : Adam
# Preferred Loss : Charbonnier loss
# Number of epochs : ___ 

import torch
import torch.nn as nn
import time
import os

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        return torch.mean(torch.sqrt((y_pred - y_true) ** 2 + self.epsilon ** 2))


def compute_metric(y_pred, y_true):
    """
    Computes PSNR (Peak Signal-to-Noise Ratio) between predicted and ground truth images.
    
    Parameters:
        y_pred (torch.Tensor): Predicted images.
        y_true (torch.Tensor): Ground truth images.
        
    Returns:
        float: PSNR value.
    """
    mse = torch.mean((y_pred - y_true) ** 2)
    if mse == 0:  # Avoid division by zero
        return float('inf')
    
    # Use the maximum pixel value for normalization (assuming images are in [0, 1])
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def train_model(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    loss_function, 
    device, 
    epochs, 
    patience, 
    val_interval, 
    lr_scheduler, 
    output_dir
):
    training = True
    best_metric = -1
    best_metric_epoch = -1
    not_improved_epoch = 0
    epoch_loss_values = []
    metric_values = []
    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        # Training Loop
        for step, batch_data in enumerate(train_loader):
            step_start = time.time()
            lr_inputs, hr_targets = batch_data["low_res"].to(device), batch_data["high_res"].to(device)

            optimizer.zero_grad()
            outputs = model(lr_inputs)
            loss = loss_function(outputs, hr_targets)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(
                f"Step {step + 1}/{len(train_loader)}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f} sec"
            )

        lr_scheduler.step()
        epoch_loss /= len(train_loader)
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation at intervals
        if (epoch + 1) % val_interval == 0:
            model.eval()
            metric = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_lr_inputs, val_hr_targets = val_data["low_res"].to(device), val_data["high_res"].to(device)
                    val_outputs = model(val_lr_inputs)
                    metric += compute_metric(val_outputs, val_hr_targets)

                metric /= len(val_loader)
                metric_values.append(metric)

                # Save best model
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    not_improved_epoch = 0
                    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                    print(f"Saved new best model at epoch {epoch + 1}")
                else:
                    not_improved_epoch += 1
                    if not_improved_epoch >= patience:
                        training = False
                        print("Early stopping as model performance hasn't improved")
                        break

            print(
                f"Epoch {epoch + 1} validation metric: {metric:.4f},"
                f" best metric so far: {best_metric:.4f} at epoch {best_metric_epoch}"
            )

        print(f"Time taken for epoch {epoch + 1}: {(time.time() - epoch_start):.4f} seconds")

        if not training:
            print("Training stopped early due to no improvement.")
            break

    total_time = time.time() - total_start
    print(f"Total training time: {total_time:.4f} seconds")