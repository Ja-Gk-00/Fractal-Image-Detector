import numpy as np

from fractal_detector.config import FeatureConfig
from fractal_detector.io import FloatArray
from fractal_detector.kernels import affine_match


def transform(block: FloatArray, transform_id: int) -> FloatArray:
    match transform_id:
        case 0:
            return block
        case 1:
            return np.rot90(block, -1)
        case 2:
            return np.rot90(block, 2)
        case 3:
            return np.rot90(block, 1)
        case 4:
            return np.fliplr(block)
        case 5:
            return np.rot90(block.T, 2)
        case 6:
            return np.flipud(block)
        case 7:
            return block.T
        case _:
            raise ValueError("transform_id must be in 0..7")


def compression_features(
    image: FloatArray, config: FeatureConfig, backend: str
) -> tuple[dict[str, float], dict[str, object]]:
    size, side = config.block_size, image.shape[0]
    positions = [
        (y, x)
        for y in range(0, side - 2 * size + 1, config.stride)
        for x in range(0, side - 2 * size + 1, config.stride)
    ]
    if len(positions) > config.max_domains:
        indices = np.linspace(0, len(positions) - 1, config.max_domains, dtype=int)
        positions = [positions[int(i)] for i in indices]
    domains = []
    for y, x in positions:
        block = (
            image[y : y + 2 * size, x : x + 2 * size].reshape(size, 2, size, 2).mean(axis=(1, 3))
        )
        domains.extend(transform(block, t).ravel() for t in config.transform_ids)
    ranges = (
        image.reshape(side // size, size, side // size, size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, size * size)
    )
    matches = affine_match(ranges, np.asarray(domains), config.s_clip, backend)
    errors, contrasts = matches[:, 0], np.abs(matches[:, 1])
    return {
        "compression_mse_mean": float(errors.mean()),
        "compression_mse_std": float(errors.std()),
        "compression_mse_p90": float(np.quantile(errors, 0.9)),
        "compression_contrast_mean": float(contrasts.mean()),
        "compression_contrast_std": float(contrasts.std()),
    }, {
        "domain_positions": positions,
        "transform_ids": config.transform_ids,
        "matches_mse_contrast_offset_domain": matches.tolist(),
        "range_grid_side": side // size,
    }
