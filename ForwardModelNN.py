from torch import nn;
import torch;


class FeedForwardNN(nn.Module):  # Renamed the class for clarity
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()  # Fixed super call

        # Define layers
        self.linear_stack = nn.Sequential(
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape),
        )

    def forward(self, x: torch.Tensor):
        return self.linear_stack(x)