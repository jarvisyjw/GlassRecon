"""Preserve original TUM poses and omit ambiguous repeated timestamp samples."""

import hashlib
from collections import Counter
from pathlib import Path

from experiments.run_comparison import save_json


def prepare_poses(source, destination):
    source, destination = Path(source), Path(destination)
    text = source.read_text()
    rows = [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    counts = Counter(float(line.split()[0]) for line in rows)
    duplicated = {stamp for stamp, count in counts.items() if count > 1}
    if not duplicated:
        return source
    retained = [line for line in rows if float(line.split()[0]) not in duplicated]
    prepared = "# TUM camera-to-world poses; all repeated timestamp samples omitted\n"
    prepared += "\n".join(retained) + "\n"
    destination.mkdir(parents=True, exist_ok=True)
    result = destination / "groundtruth_unique.txt"
    if result.exists() and result.read_text() != prepared:
        raise ValueError(f"Refusing to replace different prepared poses: {result}")
    result.write_text(prepared)
    save_json(
        destination / "pose_preparation.json",
        {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "prepared": str(result),
            "prepared_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
            "policy": "Omit every sample at a repeated timestamp; preserve other original rows.",
            "input_rows": len(rows),
            "retained_rows": len(retained),
            "removed_timestamps": sorted(duplicated),
            "removed_rows": [line for line in rows if float(line.split()[0]) in duplicated],
            "interpolation": "Unchanged PoseTrajectory interpolation and maximum-gap check.",
        },
    )
    return result
