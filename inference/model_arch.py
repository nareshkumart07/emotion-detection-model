import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES


class EEGBiLSTMClassifier(nn.Module):
    """BiLSTM over EEG channels; each step is a 4-dim band-power vector."""

    def __init__(
        self,
        input_size=4,
        hidden_size=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.5,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers - 1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 2 else 0.0,
        )
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        return self.fc(out)


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al., 2018) for input shaped (batch, channels, samples)."""

    def __init__(
        self,
        n_channels: int = 14,
        n_samples: int = 256,
        n_classes: int = 2,
        dropout: float = 0.5,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * d)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(
            f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False
        )
        self.conv4 = nn.Conv2d(f1 * d, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        self._flatten_size = self._infer_flatten_size(n_channels, n_samples)
        self.fc = nn.Linear(self._flatten_size, n_classes)

    def _infer_flatten_size(self, n_channels: int, n_samples: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            x = self._features(x)
            return x.reshape(1, -1).shape[1]

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
