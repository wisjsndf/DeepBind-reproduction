import torch
import torch.nn as nn

class DeepBind(nn.Module):
    def __init__(
        self,
        num_kernels: int = 16,
        kernel_size: int = 24,
        fc_hidden: int = 32,
        dropout_rate: float = 0.0,
        conv_bias: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=num_kernels,
            kernel_size=kernel_size,
            bias=True
        )
        nn.init.constant_(self.conv.bias, conv_bias)
        self.relu = nn.ReLU()
        if fc_hidden and int(fc_hidden) > 0:
            self.fc = nn.Sequential(
                nn.Linear(2 * num_kernels, int(fc_hidden)),
                nn.ReLU(),
                nn.Dropout(p=float(dropout_rate)) if dropout_rate and float(dropout_rate) > 0.0 else nn.Identity(),
                nn.Linear(int(fc_hidden), 1)
            )
        else:
            self.fc = nn.Linear(2 * num_kernels, 1)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.relu(h)
        h_max = h.amax(dim=2)
        h_avg = h.mean(dim=2)
        h_cat = torch.cat([h_max, h_avg], dim=1)
        out = self.fc(h_cat).squeeze(1)
        return out
        