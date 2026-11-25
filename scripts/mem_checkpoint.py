"""Fetch the public pi05_libero checkpoint and verify each GCS object."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
from urllib.parse import quote

import google_crc32c


ROOT = Path(__file__).resolve().parents[1] / "data/mem/checkpoints"
PREFIX = "checkpoints/pi05_libero/"
BUCKET = "https://storage.googleapis.com"


def fetch_range(job):
    url, offset, end, chunk = job
    headers = subprocess.check_output(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "20",
                                       "--max-time", "240", "--range", f"{offset}-{end}",
                                       "-D", "-", "-o", str(chunk), url]).decode()
    if (f"bytes {offset}-{end}/" not in headers.lower() or chunk.stat().st_size != end-offset+1):
        raise RuntimeError(f"Invalid range response at {offset}")
    return chunk


def download(item):
    key = item["name"]
    if not key.startswith(PREFIX) or key.endswith("/"):
        return
    target = ROOT / "pi05_libero" / key.removeprefix(PREFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    size = int(item["size"])
    if not target.exists() and (not partial.exists() or partial.stat().st_size != size):
        url = f"{BUCKET}/openpi-assets/{quote(key)}?generation={item['generation']}"
        offset = partial.stat().st_size if partial.exists() else 0
        if size < 8 << 20:
            subprocess.run(["curl", "-fsSL", "--retry", "3", url, "-o", str(partial)], check=True)
        else:
            block = 16 << 20
            with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
                jobs = ((url + f"&offset={start}", start, min(start+block, size)-1, Path(scratch)/str(start))
                        for start in range(offset, size, block))
                with ThreadPoolExecutor(max_workers=8) as pool, partial.open("ab") as dest:
                    for chunk in pool.map(fetch_range, jobs):
                        with chunk.open("rb") as src:
                            shutil.copyfileobj(src, dest)
                        chunk.unlink()
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
    for item in listing["items"]:
        download(item)
    (ROOT / "manifest.json").write_text(json.dumps(listing["items"], indent=2))
