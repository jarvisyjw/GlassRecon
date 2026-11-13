import json

import pytest

from experiments.prepare_tum_poses import prepare_poses
from glassrecon.data import PoseTrajectory


def test_ambiguous_pose_timestamp_is_omitted_and_original_is_preserved(tmp_path):
    source = tmp_path / "original.txt"
    text = "# poses\n0 0 0 0 0 0 0 1\n1.000 1 0 0 0 0 0 1\n1.0 2 0 0 0 0 0 1\n2 3 0 0 0 0 0 1\n"
    source.write_text(text)
    with pytest.raises(ValueError, match="timestamps must be unique"):
        PoseTrajectory(source)
    result = prepare_poses(source, tmp_path / "prepared")
    trajectory = PoseTrajectory(result)
    assert trajectory.times.tolist() == [0, 2]
    assert trajectory.at(1, max_difference=0.05) is None
    assert source.read_text() == text
    receipt = json.loads((result.parent / "pose_preparation.json").read_text())
    assert receipt["removed_timestamps"] == [1.0]
    assert len(receipt["removed_rows"]) == 2
    assert prepare_poses(source, result.parent) == result
    assert prepare_poses(result, tmp_path / "unused") == result
    assert not (tmp_path / "unused").exists()
