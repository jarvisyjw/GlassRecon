import json
from concurrent.futures import Future
from contextlib import nullcontext
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from glassrecon import live
from glassrecon.cli import build_parser
from glassrecon.orb_stream import OrbPose


def test_replay_starts_fresh_run_and_preserves_original_output(tmp_path, monkeypatch):
    viewer = Mock()
    output = tmp_path / "office"
    args = SimpleNamespace(output=str(output), autoplay=False, exit_after_run=False)
    runs = []

    def replay(settings, observer):
        assert observer is viewer
        runs.append((settings.output, settings.autoplay))
        return {"interrupted": len(runs) == 2}

    monkeypatch.setattr(live, "run_live", replay)
    live.serve_live(args, viewer)
    assert runs == [(str(output), False), (str(output / "replays/run_002"), True)]
    viewer.wait_for_replay.assert_called_once_with()
    assert args.output == str(output) and not args.autoplay


def prepare_replay(tmp_path, monkeypatch, segment_ids, before_track, predict):
    # The public dataset contains only calibration, RGB and raw depth.
    dataset = tmp_path / "rgbd"
    for folder in ("rgb", "depth"):
        (dataset / folder).mkdir(parents=True)
    width, height = 32, 24
    camera = dict(width=width, height=height, fx=30, fy=30, cx=15.5, cy=11.5)
    (dataset / "intrinsics.json").write_text(json.dumps(camera))
    xx = np.indices((height, width))[1].astype(np.uint8)
    raw = 1000 + xx.astype(np.uint16) * 30
    raw[7, 5] = 0
    packets = []
    for index, segment in enumerate(segment_ids):
        key = f"frame_{index:02d}"
        stamp = 10.0 + 0.01 * index
        rgb = np.stack([xx, np.zeros_like(xx), np.full_like(xx, index)], axis=-1)
        Image.fromarray(rgb).save(dataset / "rgb" / f"{key}.png")
        Image.fromarray(raw).save(dataset / "depth" / f"{key}.png")
        pose = None
        if segment is not None:
            pose = np.eye(4)
            pose[0, 3] = 4.0 * segment
        packets.append(
            OrbPose(index, stamp, key, pose, segment, "OK" if pose is not None else "LOST")
        )
    for folder in ("rgb", "depth"):
        (dataset / f"{folder}.txt").write_text(
            "".join(f"{p.timestamp} {folder}/{p.key}.png\n" for p in packets)
        )

    def stream():
        for packet in packets:
            before_track(packet.index)
            yield packet

    monkeypatch.setattr(live, "open_orb_poses", lambda *args, **kwargs: nullcontext(stream()))
    predictor = Mock()
    predictor.predict.side_effect = predict
    monkeypatch.setattr(live, "DA3Worker", lambda *args, **kwargs: nullcontext(predictor))
    vocabulary = tmp_path / "ORBvoc.txt"
    vocabulary.touch()
    args = build_parser().parse_args(
        [
            "live",
            "--dataset",
            str(dataset),
            "--output",
            str(tmp_path / "out"),
            "--orb-python",
            "test-orb",
            "--orb-vocabulary",
            str(vocabulary),
            "--model-python",
            "test-da3",
            "--depth-units-per-meter",
            "1000",
            "--alignment-iterations",
            "1",
            "--autoplay",
        ]
    )
    return args, packets


def test_async_tracking_queue_drops_and_segment_flush(tmp_path, monkeypatch):
    # The first six tracked frames reach the mapper. While its sixth prediction is
    # blocked, segment 1 fills and overflows the queue. Release on the last lost frame.
    segments = [0] * 6 + [None] + [1] * 5 + [None]
    started = [Event() for _ in range(6)]
    release = Event()
    predicted = []

    def before_track(index):
        if 1 <= index <= 6:
            assert started[index - 1].wait(5)
        if index == len(segments) - 1:
            release.set()

    def predict(rgb, key):
        index = int(rgb[0, 0, 2])
        assert key == f"frame_{index:02d}"
        predicted.append(index)
        if index < len(started):
            started[index].set()
        if index == 5:
            assert release.wait(5), "Tracking stalled behind depth inference"
        return 1.0 + rgb[:, :, 0].astype(np.float32) * 0.05

    args, packets = prepare_replay(tmp_path, monkeypatch, segments, before_track, predict)
    integrated = []

    def create_mapper(camera, backend):
        assert backend == "nvblox"
        frames = []
        segment = len(integrated)
        integrated.append(frames)
        mapper = Mock(integrated=0)
        mapper.metadata.return_value = {"backend": backend}
        mapper.point_cloud.return_value = (np.ones((1, 3)), np.ones((1, 3)))
        mapper.export.return_value = {}

        def integrate(rgb, depth, pose):
            # Real alignment and five-frame recovery must fill the raw sensor hole.
            np.testing.assert_allclose(depth, 1.0 + rgb[:, :, 0] * 0.03, atol=1e-4)
            assert pose[0, 3] == 4.0 * segment
            frames.append(int(rgb[0, 0, 2]))
            mapper.integrated += 1

        mapper.integrate.side_effect = integrate
        return mapper

    monkeypatch.setattr(live, "create_mapper", create_mapper)
    viewer = Mock()
    summary = live.run_live(args, viewer)
    assert summary["processed"] == len(segments)
    assert summary["tracking_unavailable"] == 2
    assert summary["mapping_queue_dropped"] == 3
    assert summary["alignment_failed"] == 0
    assert summary["integrated"] == 8
    assert summary["tracked"] == 11
    assert predicted == [0, 1, 2, 3, 4, 5, 10, 11]
    assert integrated == [list(range(6)), [10, 11]]

    first_cloud = next(i for i, call in enumerate(viewer.mock_calls) if call[0] == "update_map")
    assert sum(call[0] == "on_tracking" for call in viewer.mock_calls[:first_cloud]) >= 5
    assert [call.args[1] for call in viewer.on_tracking.call_args_list] == [
        packet.timestamp for packet in packets
    ]
    assert 0 <= summary["first_pose_seconds"] < summary["first_cloud_seconds"]
    assert [s["tracked"] for s in summary["segments"]] == [6, 5]
    assert [s["integrated"] for s in summary["segments"]] == [6, 2]
    for index, frames in enumerate(integrated):
        output = tmp_path / "out" / f"segment_{index:03d}"
        tracked = np.loadtxt(output / "trajectory.txt", ndmin=2)
        fused = np.loadtxt(output / "fused_trajectory.txt", ndmin=2)
        np.testing.assert_array_equal(
            tracked[:, 0], [p.timestamp for p in packets if p.map_id == index]
        )
        np.testing.assert_array_equal(fused[:, 0], [packets[i].timestamp for i in frames])
        np.testing.assert_array_equal(fused[:, 1], 4.0 * index)


def test_mapper_error_with_full_queue_reaches_caller(tmp_path, monkeypatch):
    started, release = Event(), Event()

    def before_track(index):
        if index == 1:
            assert started.wait(5)
        if index == 4:
            release.set()

    def predict(rgb, key):
        started.set()
        assert release.wait(5)
        raise RuntimeError("DA3 prediction failed at frame_00")

    args, _ = prepare_replay(tmp_path, monkeypatch, [0, 0, 0, 0, None], before_track, predict)
    mapper = Mock()
    mapper.metadata.return_value = {}
    monkeypatch.setattr(live, "create_mapper", lambda *args, **kwargs: mapper)
    outcome = Future()

    def replay():
        try:
            outcome.set_result(live.run_live(args, Mock()))
        except Exception as exc:
            outcome.set_exception(exc)

    # A regression that blocks the EOF sentinel on the full queue must fail this
    # test within a bounded interval, even after the mapping worker has exited.
    Thread(target=replay, daemon=True).start()
    with pytest.raises(RuntimeError, match="DA3 prediction failed at frame_00"):
        outcome.result(timeout=10)
