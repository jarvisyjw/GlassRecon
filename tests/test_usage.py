"""Regression checks for prepared-depth and cross-environment entry points."""

import subprocess
from unittest.mock import patch

import pytest

from examples import run_full_pipeline
from glassrecon.cli import main


def test_metric_depth_requires_meter_units():
    with pytest.raises(SystemExit) as error:
        main(
            [
                "reconstruct",
                "--dataset",
                "unused",
                "--output",
                "unused",
                "--depth-source",
                "metric",
                "--pose-source",
                "file",
                "--pose-file",
                "unused",
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("provided_pose", [True, False])
def test_full_pipeline_stage_contract(tmp_path, provided_pose):
    dataset = tmp_path / "input"
    dataset.mkdir()
    for name in ["rgb.txt", "depth.txt", "intrinsics.json", "poses.txt"]:
        (dataset / name).touch()
    (dataset / "rgb").mkdir()
    if provided_pose:
        (dataset / "masks").mkdir()
    argv = [
        "runner",
        "--dataset",
        str(dataset),
        "--model-python",
        "/model/python",
        "--mapping-python",
        "/mapping/python",
        "--output",
        str(tmp_path / "output"),
    ]
    if provided_pose:
        argv += ["--poses", str(dataset / "poses.txt")]
    else:
        argv += [
            "--rgb-folder",
            str(dataset / "rgb"),
            "--intrinsics",
            str(dataset / "intrinsics.json"),
        ]
    with patch("sys.argv", argv), patch.object(subprocess, "run") as run:
        run_full_pipeline.main()
    predict, mapping = [call.args[0] for call in run.call_args_list]
    assert predict[0] == "/model/python"
    assert predict[predict.index("--image-folder") + 1] == str(dataset / "rgb")
    assert predict[predict.index("--model-id") + 1] == "depth-anything/DA3-LARGE"
    assert mapping[0] == "/mapping/python"
    assert mapping[mapping.index("--backend") + 1] == "nvblox"
    assert mapping[mapping.index("--intrinsics") + 1] == str(dataset / "intrinsics.json")
    if provided_pose:
        assert mapping[mapping.index("--valid-mask-dir") + 1] == str(dataset / "masks")
    else:
        assert "--valid-mask-dir" not in mapping
    assert mapping[mapping.index("--prediction-dir") + 1] == predict[-1]
    assert mapping[mapping.index("--pose-source") + 1] == ("file" if provided_pose else "odometry")
    with (
        patch("sys.argv", argv),
        patch.object(
            subprocess, "run", side_effect=subprocess.CalledProcessError(1, "predict")
        ) as run,
    ):
        with pytest.raises(subprocess.CalledProcessError):
            run_full_pipeline.main()
        assert run.call_count == 1
