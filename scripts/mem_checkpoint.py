"""Fetch the public pi05_libero checkpoint and verify each GCS object."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

import google_crc32c


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
    size = int(item["size"])
    if not target.exists() and (not partial.exists() or partial.stat().st_size != size):
        subprocess.run(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "20",
                        "--speed-limit", "1024", "--speed-time", "90", "-C", "-",
                        f"{BUCKET}/openpi-assets/{quote(key)}?generation={item['generation']}",
                        "-o", str(partial)], check=True)
    source = target if target.exists() else partial
    checksum = hashlib.md5() if "md5Hash" in item else google_crc32c.Checksum()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(block)
    digest = base64.b64encode(checksum.digest()).decode()
    if digest != item.get("md5Hash", item["crc32c"]) or source.stat().st_size != size:
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
