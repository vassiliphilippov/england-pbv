"""Build a comparison gallery for a next-gen render round into ~/temp/nextgen-pairs-<tag>.

Run: uv run python -m england_pbv.site.gallery <tag> "<what changed>" ["<known issues>"]
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from england_pbv import paths
from england_pbv.terrain.grid import latlon_to_bng

ORDER: tuple[str, ...] = ("devilsdyke", "wensleydale", "lawley", "cheddar", "raggedstone")


def main() -> None:
    tag = sys.argv[1]
    changed = sys.argv[2]
    issues = sys.argv[3] if len(sys.argv) > 3 else ""
    out = Path.home() / "temp" / f"nextgen-pairs-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    pairs = paths.OUTPUTS_DIR / "calibration" / "pairs_nextgen"
    for jpg in pairs.glob("pair_*_n1.jpg"):
        shutil.copy(jpg, out / jpg.name)
    timings = json.loads((pairs / "timings.json").read_text(encoding="utf-8"))
    sites = {
        s["key"]: s
        for s in json.loads((paths.OUTPUTS_DIR / "nextgen_sites.json").read_text(encoding="utf-8"))
    }
    sat = np.load(paths.SATELLITE10_GRID_NPY, mmap_mode="r")
    figs = []
    for key in ORDER:
        site = sites[key]
        bng = latlon_to_bng(float(site["lat"]), float(site["lon"]))
        col = int((bng.easting - 80000) / 10)
        row = int((bng.northing - 4000) / 10)
        window = np.array(sat[row - 200 : row + 200, col - 200 : col + 200], dtype=np.uint8)
        covered = float((window.sum(axis=2) > 12).mean())
        figs.append(
            f'<section><h2>{site["name"]}</h2><p class="meta">bearing {site["bearing"]}&deg; '
            f"&middot; {site['focal35']} mm &middot; render {timings[key]['new_s']:.2f} s "
            f"&middot; satellite {covered:.0%}</p>"
            f'<img src="pair_{key}_n1.jpg"></section>'
        )
    versions = ("r1", "v2", "v3", "v4", "v5", "v6", "v7", "v8")
    link_parts = []
    for version in versions:
        folder = "nextgen-pairs" if version == "r1" else f"nextgen-pairs-{version}"
        if (Path.home() / "temp" / folder).exists():
            link_parts.append(f'<a href="../{folder}/index.html">{version}</a>')
    links = " &middot; ".join(link_parts)
    issues_html = f'<h2>Known issues</h2><p class="note">{issues}</p>' if issues else ""
    (out / "index.html").write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Next-Gen Renderer — {tag}</title><style>
body{{font:15px/1.6 -apple-system,sans-serif;background:#14181c;
color:#e8e8e2;margin:0;padding:24px}}
.wrap{{max-width:1140px;margin:0 auto}} h1{{font-size:26px;margin:0 0 4px}}
h2{{font-size:19px;margin:30px 0 2px;color:#9fdcb4}}
.meta{{color:#9aa4ae;font-size:13px;margin:2px 0 10px}}
img{{width:100%;border-radius:10px;display:block}} .note{{color:#9aa4ae;font-size:13px}}
a{{color:#8fc4ea}} code{{background:#20262c;padding:1px 5px;border-radius:4px}}</style></head>
<body><div class="wrap"><h1>Next-generation renderer — {tag}</h1>
<p class="note">Rounds: {links}</p>
<h2>What changed</h2><p class="note">{changed}</p>
{issues_html}
{"".join(figs)}
<p class="note" style="margin-top:26px">EA 1 m LiDAR DTM + first-return DSM (OGL) &middot; OSM (ODbL)
&middot; contains modified Copernicus Sentinel data &middot; textures and sky Poly Haven (CC0).</p>
</div></body></html>""",
        encoding="utf-8",
    )
    subprocess.run(["open", str(out / "index.html")], check=False)
    print(f"gallery: {out}/index.html")


if __name__ == "__main__":
    main()
