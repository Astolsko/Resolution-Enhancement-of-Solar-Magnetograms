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
    mse = torch.mean((y_pred - y_true) ** 2)
    if mse == 0:  # Avoid division by zero
        return float('inf')
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
    val_loss_values = []
    metric_values = []
    total_start = time.time()

    # Define pixel-to-pixel loss (L1 Loss for this case)
    pixel_loss_function = nn.L1Loss()  # You can also use nn.MSELoss() for L2 loss

    for epoch in range(epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0

        # Training Loop
        for step, batch_data in enumerate(train_loader):
            step_start = time.time()
            lr_inputs, hr_targets = batch_data["image"].to(device), batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(lr_inputs)
            
            # Compute the Charbonnier Loss
            charbonnier_loss = loss_function(outputs, hr_targets)
            # Compute the Pixel-to-Pixel (L1) Loss
            pixel_loss = pixel_loss_function(outputs, hr_targets)
            # Combine the losses (you can adjust the weight for each loss here)
            combined_loss = pixel_loss + charbonnier_loss

            combined_loss.backward()

            # Calculate the gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5  # L2 norm

            # Update the weights
            optimizer.step()

            epoch_loss += combined_loss.item()

            # Print the current loss and gradient norm for debugging
            print(f"Step {step + 1}/{len(train_loader)}, Charbonnier Loss: {charbonnier_loss.item():.4f}, "
                  f"Pixel Loss: {pixel_loss.item():.4f}, Combined Loss: {combined_loss.item():.4f}, "
                  f"Gradient Norm: {total_norm:.4f}, Step time: {(time.time() - step_start):.4f} sec")

        lr_scheduler.step()
        epoch_loss /= len(train_loader)
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation at intervals
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            metric = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_lr_inputs, val_hr_targets = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = model(val_lr_inputs)
                    val_loss += loss_function(val_outputs, val_hr_targets).item()
                    metric += compute_metric(val_outputs, val_hr_targets)

                val_loss /= len(val_loader)
                val_loss_values.append(val_loss)
                metric /= len(val_loader)
                metric_values.append(metric)

                print(f"Validation Loss: {val_loss:.4f}")
                print(f"Validation PSNR: {metric:.4f}")

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
    print(f"All epoch losses: {epoch_loss_values}")
    print(f"Validation losses: {val_loss_values}")
    print(f"PSNR metrics: {metric_values}")
