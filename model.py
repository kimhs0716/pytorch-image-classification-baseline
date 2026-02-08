import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        # (1, 28, 28)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1) 
        # (32, 28, 28)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (32, 14, 14)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # (64, 14, 14)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (64, 7, 7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_model(model_type: str) -> nn.Module:
    """Build and return model."""
    model_type = model_type.strip().lower()
    if model_type == "cnn":
        model = SimpleCNN(1, 10)
    elif model_type == "mlp":
        model = SimpleMLP(28 * 28, 128, 10)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

