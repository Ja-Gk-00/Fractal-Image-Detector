import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fractal_detector.config import NeuralConfig


def quadrants(x: Tensor) -> Tensor:
    height, width = x.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError("spectral quadrant splitting requires even dimensions")
    h, w = height // 2, width // 2
    return torch.stack((x[..., :h, :w], x[..., :h, w:], x[..., h:, :w], x[..., h:, w:]))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x + self.layers(x), negative_slope=0.2)


class DnCNN(nn.Module):
    def __init__(self, depth: int, channels: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers.extend(
                (nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
            )
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class LearnedFractalUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        parts = quadrants(x)
        transformed = self.branch(parts.flatten(0, 1)).unflatten(0, (4, x.shape[0]))
        fused = self.fusion(transformed.prod(dim=0))
        return fused.mean(dim=(-2, -1)), transformed.mean(dim=0)


class FractalNetwork(nn.Module):
    srm: Tensor

    def __init__(self, config: NeuralConfig) -> None:
        super().__init__()
        self.config = config
        c = config.channels
        self.denoiser = (
            DnCNN(config.dncnn_depth, config.dncnn_channels)
            if config.architecture == "residual_fractal" and config.residual == "dncnn"
            else nn.Identity()
        )
        kernels = (
            torch.tensor(
                [
                    [
                        [0, 0, 0, 0, 0],
                        [0, -1, 2, -1, 0],
                        [0, 2, -4, 2, 0],
                        [0, -1, 2, -1, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [
                        [-1, 2, -2, 2, -1],
                        [2, -6, 8, -6, 2],
                        [-2, 8, -12, 8, -2],
                        [2, -6, 8, -6, 2],
                        [-1, 2, -2, 2, -1],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 1, -2, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                ],
                dtype=torch.float32,
            )
            / torch.tensor([4, 12, 2])[:, None, None]
        )
        self.register_buffer("srm", kernels[:, None])
        if config.architecture == "srm_fractal":
            self.backbone = nn.Sequential(
                nn.Conv2d(3, c, 1),
                nn.BatchNorm2d(c),
                nn.LeakyReLU(0.2),
                *(ResidualBlock(c) for _ in range(config.backbone_blocks)),
                nn.Conv2d(c, c, 3, padding=1),
            )
            self.spectral_norm = nn.BatchNorm2d(c)
            self.units = nn.ModuleList()
            self.embedding = nn.Identity()
            self.head = nn.Linear(c * config.levels, 1)
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(1, c, 3, padding=1),
                nn.InstanceNorm2d(c),
                nn.LeakyReLU(0.2),
                nn.Conv2d(c, c, 3, padding=1),
                nn.InstanceNorm2d(c),
                nn.LeakyReLU(0.2),
            )
            self.spectral_norm = nn.Identity()
            self.units = nn.ModuleList(LearnedFractalUnit(c) for _ in range(config.levels))
            self.embedding = nn.Sequential(
                nn.Linear(c * config.levels, 128), nn.LeakyReLU(0.2), nn.Dropout(config.dropout)
            )
            self.head = nn.Linear(128, 1)

    def residual(self, x: Tensor) -> Tensor:
        config = self.config
        if config.residual == "dncnn":
            return self.denoiser(x)
        k = config.median_size
        if config.residual == "median":
            patches = F.pad(x, (k // 2,) * 4, mode="reflect").unfold(2, k, 1).unfold(3, k, 1)
            return x - patches.flatten(-2).median(dim=-1).values
        axis = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
        weights = torch.exp(-axis.square() / (2 * config.gaussian_sigma**2))
        kernel = weights[:, None] * weights[None, :]
        kernel = kernel / kernel.sum()
        return x - F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), kernel[None, None])

    def encode(self, x: Tensor) -> Tensor:
        config = self.config
        if config.architecture == "srm_fractal":
            spatial = self.backbone(F.conv2d(x, self.srm, padding=2))
            spectrum = torch.fft.fft2(spatial.float()).abs()
            current = self.spectral_norm(spectrum)
            levels = []
            for _ in range(config.levels):
                parts = quadrants(current)
                levels.append(parts.prod(dim=0).mean(dim=(-2, -1)))
                current = parts.mean(dim=0)
        else:
            spectrum = torch.fft.fftshift(
                torch.fft.fft2(self.residual(x).float()), dim=(-2, -1)
            ).abs()
            spectrum = spectrum / spectrum.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            current = self.backbone(spectrum)
            levels = []
            for index, unit in enumerate(self.units):
                pooled, current = unit(current)
                levels.append(pooled)
                if index + 1 < config.levels:
                    current = F.avg_pool2d(current, 2)
        return self.embedding(torch.cat(levels, dim=1))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x)).squeeze(-1)
