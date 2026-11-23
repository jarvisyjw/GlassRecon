"""Download the official comparison datasets, resume verified HTTP byte ranges.

Run with Python 3.12 (uses tarfile's data extraction filter).
"""

import argparse
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TUM = "https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/"
ICL = "https://www.doc.ic.ac.uk/~ahanda/"
SOURCES = {
    "tum_desk": (TUM + "rgbd_dataset_freiburg1_desk.tgz", "data"),
    "tum_room": (TUM + "rgbd_dataset_freiburg1_room.tgz", "data"),
    "tum_desk2": (TUM + "rgbd_dataset_freiburg1_desk2.tgz", "data"),
    "tum_plant": (TUM + "rgbd_dataset_freiburg1_plant.tgz", "data"),
    "tum_fr2_desk": (
        "https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz",
        "data",
    ),
    "tum_fr3_office": (
        "https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg3/"
        "rgbd_dataset_freiburg3_long_office_household.tgz",
        "data",
    ),
    "icl_kt2_clean": (ICL + "living_room_traj2_frei_png.tar.gz", "data/icl_kt2_clean"),
    "icl_kt2_noisy": (ICL + "living_room_traj2n_frei_png.tar.gz", "data/icl_kt2_noisy"),
    "icl_mesh": (ICL + "VaFRIC/living_room_obj_mtl.tar.gz", "data/icl_mesh"),
}


def download(name, workers=4):
    url, destination = SOURCES[name]
    archive = Path("data/downloads") / url.rsplit("/", 1)[1]
    archive.parent.mkdir(parents=True, exist_ok=True)
    receipt = archive.with_name(archive.name + ".json")
    extracted = (
        Path(destination) / archive.name.removesuffix(".tgz")
        if name.startswith("tum_")
        else Path(destination)
    )
    if receipt.exists() and extracted.exists():
        print(f"{name}: already downloaded and extracted", flush=True)
        return
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith("bytes 0-0/"):
            raise RuntimeError(f"{url}: server does not support validated range downloads")
        total = int(content_range.split("/")[-1])
        etag = response.headers.get("ETag")
    parts = archive.with_name(archive.name + ".parts")
    parts.mkdir(exist_ok=True)
    identity = parts / "source.json"
    source = {"url": url, "bytes": total, "etag": etag}
    if identity.exists() and json.loads(identity.read_text()) != source:
        raise RuntimeError(f"Source changed; remove stale partial download {parts}")
    identity.write_text(json.dumps(source, indent=2))
    chunk = 4 * 1024 * 1024

    def fetch(i):
        begin, end = i * chunk, min((i + 1) * chunk, total) - 1
        path = parts / f"{i:05d}"
        for attempt in range(8):
            size = path.stat().st_size if path.exists() else 0
            if size == end - begin + 1:
                return
            if size > end - begin + 1:
                raise RuntimeError(f"Oversized partial file {path}")
            offset = begin + size
            try:
                headers = {"Range": f"bytes={offset}-{end}"}
                if etag:
                    headers["If-Range"] = etag
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as response:
                    expected = f"bytes {offset}-{end}/{total}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"Unexpected range for {name}: {response.headers}")
                    with path.open("ab") as out:
                        while block := response.read(65536):
                            out.write(block)
            except (OSError, TimeoutError):
                if attempt == 7:
                    raise
                time.sleep(1)
        if path.stat().st_size != end - begin + 1:
            raise RuntimeError(f"Incomplete download {path}")

    count = (total + chunk - 1) // chunk
    print(f"{name}: {total / 1e6:.1f} MB, {count} ranges", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(fetch, i) for i in range(count)]
        for done, job in enumerate(as_completed(jobs), 1):
            job.result()
            if done % 16 == 0 or done == count:
                print(f"{name}: {done}/{count} ranges", flush=True)
    digest = hashlib.sha256()
    with archive.open("wb") as out:
        for i in range(count):
            with (parts / f"{i:05d}").open("rb") as part:
                while block := part.read(1024 * 1024):
                    out.write(block)
                    digest.update(block)
    Path(destination).mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source_tar:
        source_tar.extractall(destination, filter="data")
    receipt.write_text(
        json.dumps({**source, "sha256": digest.hexdigest(), "extracted_to": destination}, indent=2)
        + "\n"
    )
    shutil.rmtree(parts)
    print(f"{name}: extracted to {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=list(SOURCES))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for name in args.names:
        download(name, args.workers)
