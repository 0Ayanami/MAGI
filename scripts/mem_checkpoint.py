"""Fetch the public pi05_libero checkpoint and verify each GCS object."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1] / "data/mem/checkpoints"
PREFIX = "checkpoints/pi05_libero/"
BUCKET = "https://storage.googleapis.com"


def download(item):
    key = item["name"]
    if not key.startswith(PREFIX) or key.endswith("/"):
        return
    target = ROOT / "pi05_libero" / key.removeprefix(PREFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if not target.exists():
        subprocess.run(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "20",
                        "--speed-limit", "1024", "--speed-time", "90", "-C", "-",
                        f"{BUCKET}/openpi-assets/{quote(key)}?generation={item['generation']}",
                        "-o", str(partial)], check=True)
    source = target if target.exists() else partial
    with source.open("rb") as stream:
        digest = base64.b64encode(hashlib.file_digest(stream, "md5").digest()).decode()
    if digest != item["md5Hash"] or source.stat().st_size != int(item["size"]):
        raise ValueError(f"Checkpoint integrity failure: {source}")
    if source == partial:
        partial.replace(target)


if __name__ == "__main__":
    url = f"{BUCKET}/storage/v1/b/openpi-assets/o?prefix={quote(PREFIX)}"
    listing = json.loads(subprocess.check_output(["curl", "-fsSL", "--retry", "3", url]))
    if listing.get("nextPageToken"):
        raise RuntimeError("Incomplete checkpoint listing")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, listing["items"]))
    (ROOT / "manifest.json").write_text(json.dumps(listing["items"], indent=2))
