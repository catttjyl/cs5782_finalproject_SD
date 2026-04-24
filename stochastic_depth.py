"""
A 110-layer ResNet with stochastic depth training on CIFAR-10
"""

import argparse
import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# ============================================================
# Residual Block with Stochastic Depth
# ============================================================

class ResBlock(nn.Module):
    """
    A single residual block: Conv-BN-ReLU-Conv-BN + skip connection.
    
    During training with stochastic depth, the entire block is randomly
    dropped (bypassed via identity) with probability (1 - survival_prob).
    
    At test time, the block output is scaled by survival_prob to match
    the expected activation magnitude during training (Eq. 5 in the paper).
    """

    def __init__(self, in_channels, out_channels, stride=1, survival_prob=1.0):
        super().__init__()
        self.survival_prob = survival_prob

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut for dimension mismatch (transitional blocks)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Following He et al.: average pooling + zero padding
            # (simpler alternative to learned projection for CIFAR)
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)

    def forward(self, x):
        identity = self.shortcut(x)

        if self.training:
            # During training: randomly drop this block
            if self.survival_prob < 1.0 and torch.rand(1).item() > self.survival_prob:
                # Block is dropped — pass through identity only (Eq. 3)
                return identity
            else:
                # Block is active — standard residual computation (Eq. 1)
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return F.relu(out + identity)
        else:
            # During testing: scale output by survival probability (Eq. 5)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(self.survival_prob * out + identity)


class ShortcutProjection(nn.Module):
    """
    Shortcut for transitional blocks where dimensions change.
    Uses average pooling + zero padding (following He et al. for CIFAR).
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        if self.stride > 1:
            x = F.avg_pool2d(x, self.stride)
        if self.in_channels != self.out_channels:
            # Zero-pad extra channels
            pad = self.out_channels - self.in_channels
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
        return x


# ============================================================
# 110-Layer ResNet with Stochastic Depth
# ============================================================

class ResNetStochasticDepth(nn.Module):
    """
    110-layer ResNet for CIFAR-10 with optional stochastic depth.
    
    Architecture: 3 groups of 18 residual blocks each (L=54 total blocks).
    Filter sizes: 16, 32, 64 for the three groups.
    
    Args:
        num_classes: Number of output classes (10 for CIFAR-10)
        num_blocks_per_group: Number of ResBlocks per group (18 for 110-layer)
        p_L: Survival probability of the last layer (0.5 default, 1.0 = no stochastic depth)
        stochastic: Whether to use stochastic depth
    """

    def __init__(self, num_classes=10, num_blocks_per_group=18, p_L=0.5, stochastic=True):
        super().__init__()

        self.num_blocks_per_group = num_blocks_per_group
        total_blocks = 3 * num_blocks_per_group  # L = 54 for 110-layer
        self.total_blocks = total_blocks

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Compute survival probabilities with linear decay (Eq. 4)
        # p_ell = 1 - (ell / L) * (1 - p_L)
        def get_survival_prob(block_idx):
            if not stochastic:
                return 1.0
            return 1.0 - (block_idx / total_blocks) * (1.0 - p_L)

        # Build three groups of residual blocks
        block_idx = 0

        # Group 1: 16 filters, 32x32 feature maps
        self.group1 = nn.ModuleList()
        for i in range(num_blocks_per_group):
            block_idx += 1
            stride = 1
            self.group1.append(ResBlock(16, 16, stride, get_survival_prob(block_idx)))

        # Group 2: 32 filters, 16x16 feature maps
        self.group2 = nn.ModuleList()
        for i in range(num_blocks_per_group):
            block_idx += 1
            stride = 2 if i == 0 else 1
            in_ch = 16 if i == 0 else 32
            self.group2.append(ResBlock(in_ch, 32, stride, get_survival_prob(block_idx)))

        # Group 3: 64 filters, 8x8 feature maps
        self.group3 = nn.ModuleList()
        for i in range(num_blocks_per_group):
            block_idx += 1
            stride = 2 if i == 0 else 1
            in_ch = 32 if i == 0 else 64
            self.group3.append(ResBlock(in_ch, 64, stride, get_survival_prob(block_idx)))

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        for block in self.group1:
            x = block(x)
        for block in self.group2:
            x = block(x)
        for block in self.group3:
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_survival_probs(self):
        """Return survival probabilities for all blocks (for logging)."""
        probs = []
        for group in [self.group1, self.group2, self.group3]:
            for block in group:
                probs.append(block.survival_prob)
        return probs


# ============================================================
# Training & Evaluation
# ============================================================

def get_cifar10_loaders(batch_size=128, num_workers=2):
    """Load CIFAR-10 with standard data augmentation (horizontal flip + 4px translation)."""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    
    # Hold out 5000 for validation (following the paper)
    train_indices = list(range(45000))
    val_indices = list(range(45000, 50000))
    
    train_subset = torch.utils.data.Subset(trainset, train_indices)
    val_subset = torch.utils.data.Subset(trainset, val_indices)

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return trainloader, valloader, testloader


def train_one_epoch(model, trainloader, optimizer, criterion, device):
    """Train for one epoch, return (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model, return (avg_loss, accuracy, error_rate)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return running_loss / total, accuracy, 100.0 - accuracy


def get_gradient_magnitude(model):
    """Get mean gradient magnitude of the first conv layer (for analysis, cf. Fig. 7)."""
    for name, param in model.named_parameters():
        if "conv1.weight" == name and param.grad is not None:
            return param.grad.abs().mean().item()
    return 0.0


def train_model(mode="stochastic", epochs=500, p_L=0.5, device="cuda"):
    """
    Full training loop for one model.
    
    Args:
        mode: "stochastic" or "constant"
        epochs: Number of training epochs (500 in paper)
        p_L: Survival probability of last layer (only used if mode="stochastic")
        device: "cuda" or "cpu"
    """
    stochastic = mode == "stochastic"
    print(f"\n{'='*60}")
    print(f"Training 110-layer ResNet with {'STOCHASTIC' if stochastic else 'CONSTANT'} depth")
    print(f"p_L = {p_L if stochastic else 'N/A'}, epochs = {epochs}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Data
    trainloader, valloader, testloader = get_cifar10_loaders(batch_size=128)

    # Model
    model = ResNetStochasticDepth(
        num_classes=10,
        num_blocks_per_group=18,
        p_L=p_L,
        stochastic=stochastic,
    ).to(device)

    # Print survival probabilities
    probs = model.get_survival_probs()
    print(f"Total ResBlocks: {len(probs)}")
    if stochastic:
        print(f"Survival probs: {probs[0]:.3f} (first) -> {probs[-1]:.3f} (last)")
        expected_depth = sum(probs)
        print(f"Expected depth during training: {expected_depth:.1f} blocks")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Optimizer (matching paper settings exactly)
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )

    # Learning rate schedule: divide by 10 at epochs 250 and 375
    def lr_schedule(epoch):
        if epoch < 250:
            return 1.0
        elif epoch < 375:
            return 0.1
        else:
            return 0.01

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    criterion = nn.CrossEntropyLoss()

    # Logging
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_error": [],
        "test_loss": [], "test_acc": [], "test_error": [],
        "grad_magnitude": [], "lr": [], "epoch_time": [],
    }

    best_val_error = 100.0
    best_test_error = 100.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device)

        # Get gradient magnitude (for analysis)
        grad_mag = get_gradient_magnitude(model)

        # Evaluate
        val_loss, val_acc, val_error = evaluate(model, valloader, criterion, device)
        test_loss, test_acc, test_error = evaluate(model, testloader, criterion, device)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_error"].append(val_error)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_error"].append(test_error)
        history["grad_magnitude"].append(grad_mag)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time"].append(epoch_time)

        # Track best model by validation error (as in the paper)
        if val_error < best_val_error:
            best_val_error = val_error
            best_test_error = test_error
            best_epoch = epoch
            torch.save(model.state_dict(), f"best_model_{mode}.pth")

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Error: {val_error:.2f}% | "
                f"Test Error: {test_error:.2f}% | "
                f"Best: {best_test_error:.2f}% (ep {best_epoch+1}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/3600:.2f} hours")
    print(f"Best test error: {best_test_error:.2f}% at epoch {best_epoch+1}")

    # Save history
    history["total_time_hours"] = total_time / 3600
    history["best_test_error"] = best_test_error
    history["best_epoch"] = best_epoch
    history["mode"] = mode

    with open(f"history_{mode}.json", "w") as f:
        json.dump(history, f)

    return history


# ============================================================
# Plotting (reproducing paper figures)
# ============================================================

def plot_comparison(hist_const, hist_stoch, save_path="results_comparison.png"):
    """
    Generate comparison plots similar to Fig. 3 and Fig. 7 in the paper.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(hist_const["test_error"]) + 1)

    # --- Plot 1: Test Error (Fig. 3 left) ---
    ax = axes[0, 0]
    ax.plot(epochs, hist_const["test_error"], "r-", alpha=0.7, label=f"Constant Depth ({hist_const['best_test_error']:.2f}%)")
    ax.plot(epochs, hist_stoch["test_error"], "b-", alpha=0.7, label=f"Stochastic Depth ({hist_stoch['best_test_error']:.2f}%)")
    ax.axhline(y=hist_const["best_test_error"], color="r", linestyle="--", alpha=0.3)
    ax.axhline(y=hist_stoch["best_test_error"], color="b", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Error (%)")
    ax.set_title("110-layer ResNet on CIFAR-10: Test Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Training Loss ---
    ax = axes[0, 1]
    ax.semilogy(epochs, hist_const["train_loss"], "r-", alpha=0.7, label="Constant Depth")
    ax.semilogy(epochs, hist_stoch["train_loss"], "b-", alpha=0.7, label="Stochastic Depth")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (log)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Gradient Magnitude (Fig. 7) ---
    ax = axes[1, 0]
    ax.plot(epochs, hist_const["grad_magnitude"], "r-", alpha=0.7, label="Constant Depth")
    ax.plot(epochs, hist_stoch["grad_magnitude"], "b-", alpha=0.7, label="Stochastic Depth")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Gradient Magnitude")
    ax.set_title("First Conv Layer Gradient Magnitude (cf. Fig. 7)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Training Time per Epoch ---
    ax = axes[1, 1]
    const_avg = np.mean(hist_const["epoch_time"])
    stoch_avg = np.mean(hist_stoch["epoch_time"])
    speedup = const_avg / stoch_avg if stoch_avg > 0 else 1.0
    bars = ax.bar(
        ["Constant\nDepth", "Stochastic\nDepth"],
        [const_avg, stoch_avg],
        color=["#EF4444", "#0891B2"],
        width=0.5,
    )
    ax.set_ylabel("Avg Time per Epoch (s)")
    ax.set_title(f"Training Speed ({speedup:.2f}x speedup)")
    for bar, val in zip(bars, [const_avg, stoch_avg]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}s", ha="center", va="bottom", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to {save_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stochastic Depth ResNet on CIFAR-10")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["stochastic", "constant", "both"],
                        help="Training mode")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs (500 in paper)")
    parser.add_argument("--p_L", type=float, default=0.5,
                        help="Survival probability of last layer")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, cpu, or auto")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run (20 epochs)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    epochs = 20 if args.quick else args.epochs

    if args.mode == "both":
        # Train both models and compare
        hist_const = train_model("constant", epochs=epochs, p_L=args.p_L, device=device)
        hist_stoch = train_model("stochastic", epochs=epochs, p_L=args.p_L, device=device)
        plot_comparison(hist_const, hist_stoch)

        print("\n" + "=" * 60)
        print("FINAL COMPARISON")
        print("=" * 60)
        print(f"Constant Depth - Best Test Error: {hist_const['best_test_error']:.2f}%")
        print(f"Stochastic Depth - Best Test Error: {hist_stoch['best_test_error']:.2f}%")
        print(f"Paper reports: Constant=6.41%, Stochastic=5.25%")
        print(f"Training time: Constant={hist_const['total_time_hours']:.2f}h, "
              f"Stochastic={hist_stoch['total_time_hours']:.2f}h")
    else:
        history = train_model(args.mode, epochs=epochs, p_L=args.p_L, device=device)
        print(f"\nBest test error: {history['best_test_error']:.2f}%")


if __name__ == "__main__":
    main()
