import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeChannels(nn.Module):
    def forward(self, x):
        return x.squeeze(2)


class SpatialDropout1D(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        batch_size, channels, _ = x.shape
        mask = torch.bernoulli(
            torch.ones(batch_size, channels, 1, device=x.device, dtype=x.dtype) * (1 - self.p)
        )
        return x * mask / (1 - self.p)


class TemporalEncoder(nn.Module):
    def __init__(self, input_size=1, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128), nn.GELU(), SpatialDropout1D(dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256), nn.GELU(), SpatialDropout1D(dropout)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(256, hidden_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(hidden_dim), nn.GELU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.squeeze = SqueezeChannels()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.fusion(self.squeeze(self.pool(x)))


class FrequencyEncoder(nn.Module):
    def __init__(self, input_size=1, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_size * 2, 64, kernel_size=8, padding='same'),
            nn.BatchNorm1d(64), nn.GELU(), SpatialDropout1D(dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128), nn.GELU(), SpatialDropout1D(dropout)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(hidden_dim), nn.GELU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.squeeze = SqueezeChannels()

    def forward(self, x):
        x_fft = torch.fft.rfft(x, dim=-1)
        x_freq = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_freq = self.conv_block1(x_freq)
        x_freq = self.conv_block2(x_freq)
        x_freq = self.conv_block3(x_freq)
        return self.squeeze(self.pool(x_freq))


class DualViewEncoder(nn.Module):
    def __init__(self, input_size=1, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(input_size, hidden_dim, dropout)
        self.frequency_encoder = FrequencyEncoder(input_size, hidden_dim, dropout)
        self.view_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, return_dual=False):
        h_t = self.temporal_encoder(x)
        h_f = self.frequency_encoder(x)
        if return_dual:
            return h_t, h_f
        return self.view_fusion(torch.cat([h_t, h_f], dim=1))


class ProtoTFModel(nn.Module):
    def __init__(self, num_classes, input_size=1, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = DualViewEncoder(input_size, hidden_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def forward_dual(self, x):
        """Returns (logits, h_fused, h_temporal, h_freq)."""
        h_t, h_f = self.encoder(x, return_dual=True)
        h_fused = self.encoder.view_fusion(torch.cat([h_t, h_f], dim=1))
        return self.classifier(h_fused), h_fused, h_t, h_f


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        B = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(B, device=z1.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


class TimeSeriesAugmentation:
    @staticmethod
    def jitter(x, sigma=0.03):
        return x + torch.randn_like(x) * sigma

    @staticmethod
    def scaling(x, sigma=0.1):
        factor = torch.randn(x.shape[0], 1, 1, device=x.device) * sigma + 1
        return x * factor

    @staticmethod
    def window_slice(x, reduce_ratio=0.9):
        B, C, L = x.shape
        tgt = int(L * reduce_ratio)
        if tgt >= L:
            return x
        starts = torch.randint(0, L - tgt, (B,))
        sliced = torch.zeros(B, C, tgt, device=x.device)
        for b in range(B):
            sliced[b] = x[b, :, starts[b]:starts[b] + tgt]
        return F.interpolate(sliced, size=L, mode='linear', align_corners=False)

    @staticmethod
    def augment(x, strength='medium'):
        if strength == 'medium':
            x = TimeSeriesAugmentation.jitter(x, sigma=0.03)
            x = TimeSeriesAugmentation.scaling(x, sigma=0.1)
            if torch.rand(1).item() > 0.5:
                x = TimeSeriesAugmentation.window_slice(x, reduce_ratio=0.9)
        return x
