import numpy as np
import pytest

from glassrecon.data import (
    PoseTrajectory,
    associate,
    matrix_quaternion,
    quaternion_matrix,
    read_index,
)
from glassrecon.mapping import accumulate_pose


def test_association_is_closest_one_to_one_and_time_ordered():
    rgb = [(1.0, "a"), (1.01, "b"), (2.0, "c")]
    depth = [(1.009, "d"), (1.019, "e"), (3.0, "f")]
    assert associate(rgb, depth, 0.02) == [(0, 1), (1, 0)]


def test_pose_interpolation_does_not_extrapolate_or_cross_gaps(tmp_path):
    path = tmp_path / "poses.txt"
    path.write_text("1 0 0 0 0 0 0 1\n1.1 2 0 0 0 0 1 0\n")
    trajectory = PoseTrajectory(path)
    pose = trajectory.at(1.05, 0.06)
    np.testing.assert_allclose(pose[:3, 3], [1, 0, 0])
    np.testing.assert_allclose(pose[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-6)
    assert trajectory.at(1.05, 0.01) is None
    assert trajectory.at(0.99) is None
    assert trajectory.at(1.11) is None
    assert trajectory.at(1.0) is not None


@pytest.mark.parametrize("q", [[0, 0, 0, 1], [0, 0, 1, 0], [1, 2, 3, 4], [1, 0, 0, 0]])
def test_quaternion_roundtrip(q):
    rotation = quaternion_matrix(q)
    np.testing.assert_allclose(quaternion_matrix(matrix_quaternion(rotation)), rotation, atol=1e-7)


def test_odometry_pose_accumulation_uses_inverse_relative_transform():
    previous = np.eye(4)
    previous[:3, :3] = quaternion_matrix([0, 0, 1, 1])
    previous[:3, 3] = [2, 3, 4]
    relative = np.eye(4)
    relative[0, 3] = -0.2
    current = accumulate_pose(previous, relative)
    np.testing.assert_allclose(current[:3, 3], [2, 3.2, 4])
    point = np.array([0.1, 0.2, 1, 1])
    np.testing.assert_allclose(current @ relative @ point, previous @ point)


def test_index_rejects_duplicates(tmp_path):
    index = tmp_path / "rgb.txt"
    index.write_text("# comment\n1 rgb/a.png\n1 rgb/b.png\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_index(index)
