"""Fetch CC0 ground textures and pack them into mip pyramids for the renderer.

Procedural noise can vary a colour but cannot reproduce the *structure* of turf —
blade-scale contrast, clumping, dead-versus-green mixing, bare patches. That
structure is what a photograph has and our foreground does not, so the renderer
needs real photo texture.

Textures are stored as a single float32 array of shape (n_textures, LEVELS, SIZE,
SIZE, 3): every mip level kept at full SIZE so the numba kernel can index a level
directly by the pixel's ground footprint (band-limiting, i.e. no aliasing) without
ragged arrays. Only the luminance *pattern* is used at render time — colour keeps
coming from the satellite mosaic — so the material stays data-driven.

Source: Poly Haven (CC0, https://polyhaven.com/license).

Run: uv run python -m england_pbv.data.ground_textures
"""

import numpy as np
import requests
from PIL import Image

from england_pbv import paths

API_FILES: str = "https://api.polyhaven.com/files/{slug}"
# green sward, dry/rough upland grass, trodden earth for paths
TEXTURE_SLUGS: tuple[str, ...] = ("aerial_grass_rock", "brown_mud_leaves_01", "aerial_rocks_02")
FALLBACK_SLUGS: tuple[str, ...] = ("forrest_ground_01", "coast_sand_rocks_02", "dirt_floor")
SIZE: int = 512
LEVELS: int = 6
USER_AGENT: str = "england-pbv-nextgen (github.com/vassiliphilippov/england-pbv)"


def _diffuse_url(slug: str) -> str | None:
    try:
        resp = requests.get(
            API_FILES.format(slug=slug), timeout=120, headers={"User-Agent": USER_AGENT}
        )
        resp.raise_for_status()
        files = resp.json()
    except (requests.RequestException, ValueError):
        return None
    for key in ("Diffuse", "diffuse", "Color", "col"):
        entry = files.get(key)
        if isinstance(entry, dict):
            for res in ("1k", "2k"):
                block = entry.get(res)
                if isinstance(block, dict):
                    jpg = block.get("jpg") or block.get("png")
                    if isinstance(jpg, dict) and "url" in jpg:
                        return str(jpg["url"])
    return None


def _pyramid(img: Image.Image) -> np.ndarray:
    """(LEVELS, SIZE, SIZE, 3) float32; level k is box-filtered by 2^k."""
    base = img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    out = np.zeros((LEVELS, SIZE, SIZE, 3), dtype=np.float32)
    for level in range(LEVELS):
        step = 2**level
        small = base.resize((max(1, SIZE // step), max(1, SIZE // step)), Image.Resampling.BOX)
        out[level] = np.asarray(
            small.resize((SIZE, SIZE), Image.Resampling.NEAREST), dtype=np.float32
        )
    return out


def main() -> None:
    out_dir = paths.OUTPUTS_DIR / "nextgen"
    out_dir.mkdir(parents=True, exist_ok=True)
    stack: list[np.ndarray] = []
    chosen: list[str] = []
    for wanted, backup in zip(TEXTURE_SLUGS, FALLBACK_SLUGS, strict=True):
        url = _diffuse_url(wanted) or _diffuse_url(backup)
        slug = wanted if _diffuse_url(wanted) else backup
        assert url is not None, f"no CC0 diffuse texture found for {wanted}/{backup}"
        raw = requests.get(url, timeout=300, headers={"User-Agent": USER_AGENT})
        raw.raise_for_status()
        path = out_dir / f"tex_{slug}.jpg"
        path.write_bytes(raw.content)
        stack.append(_pyramid(Image.open(path)))
        chosen.append(slug)
        print(f"{slug}: {url.rsplit('/', 1)[-1]}", flush=True)
    array = np.stack(stack).astype(np.float32)
    # Normalise each texture to unit mean luminance: the kernel uses it as a
    # multiplicative detail pattern over satellite colour, never as the colour.
    for i in range(array.shape[0]):
        mean = float(array[i, 0].mean())
        array[i] /= max(1.0, mean)
    np.save(out_dir / "ground_textures.npy", array)
    print(f"ground_textures.npy: {array.shape} from {chosen}")


if __name__ == "__main__":
    main()
