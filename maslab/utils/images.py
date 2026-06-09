from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImageLoadResult:
    images: list[Any]
    warnings: list[dict[str, Any]]


def resolve_image_path(raw_path: str, image_root: Path | None, source: str | None = None) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if image_root is None:
        raise FileNotFoundError(f"Image not found and no image root was provided: {raw_path}")

    filename = path.name
    candidates: list[Path] = []
    if source:
        candidates.append(image_root / source / filename)
        candidates.append(image_root / source.upper() / filename)
    candidates.append(image_root / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(image_root.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image not found under {image_root}: {raw_path}")


def load_sample_images_with_warnings(
    sample: dict[str, Any],
    image_root: Path | None,
    *,
    tolerate_truncated: bool = True,
) -> ImageLoadResult:
    image_paths = sample.get("images") or []
    if not image_paths:
        return ImageLoadResult(images=[], warnings=[])
    try:
        from PIL import Image, ImageFile
    except Exception as exc:
        raise RuntimeError("Pillow is required to load local image paths") from exc

    source = sample.get("source")
    images = []
    warnings = []
    for raw_path in image_paths:
        path = resolve_image_path(str(raw_path), image_root, source=str(source) if source else None)
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        except OSError as exc:
            if not tolerate_truncated or "truncated" not in str(exc).lower():
                raise

            previous_mode = ImageFile.LOAD_TRUNCATED_IMAGES
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous_mode

            warnings.append(
                {
                    "type": "truncated_image_tolerated",
                    "raw_path": str(raw_path),
                    "resolved_path": str(path),
                    "message": str(exc),
                    "possible_effect": "Image loaded with tolerant mode; visual content may be partially degraded.",
                }
            )
    return ImageLoadResult(images=images, warnings=warnings)


def load_sample_images(sample: dict[str, Any], image_root: Path | None) -> list[Any]:
    return load_sample_images_with_warnings(sample, image_root).images
