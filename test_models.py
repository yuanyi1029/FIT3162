import torch.nn as nn
import torch

class SimpleModel(nn.Module):
    def __init__(self, num_blocks=3):
        super(SimpleModel, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 3, kernel_size=3, padding=1)
            ) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# You might also want to move your other dummy models here
class DummyModel(nn.Module):
    """A very simple model for testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

class ComplexDummyModel(nn.Module):
    """A more complex model with blocks for testing."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 1, kernel_size=3, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=3, padding=1)
            )
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class SimplerDummyModel(nn.Module):
    """A smaller model to represent pruned result."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(4, 1, kernel_size=3, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 1, kernel_size=3, padding=1)
            )
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x