"""Pull one causal ORB-SLAM3 pose per RGB-D frame from an isolated interpreter."""

import argparse
import json
import os
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import NamedTuple

import numpy as np

from .data import CameraIntrinsics, load_sequence
from .mapping import validate_pose
from .orb_frontend import _camera_fps, _camera_settings, _track_raw_rgbd


class OrbPose(NamedTuple):
    index: int
    timestamp: float
    key: str
    pose: np.ndarray | None
    map_id: int | None  # Consecutive tracking segment, not a native Atlas ID.
    state: str


def _send(sock, packet):
    sock.sendall((json.dumps(packet, allow_nan=False) + "\n").encode())


def _receive(reader, log):
    line = reader.readline(64 * 1024)
    if not line:
        length = os.fstat(log.fileno()).st_size
        tail = os.pread(log.fileno(), 2048, max(0, length - 2048)).decode(errors="replace").strip()
        raise RuntimeError(f"ORB worker closed the pose channel unexpectedly. {tail}")
    if not line.endswith(b"\n"):
        raise RuntimeError("Incomplete ORB pose response")
    packet = json.loads(line)
    if "error" in packet:
        raise RuntimeError(f"ORB worker failed: {packet['error']}")
    return packet


def _request_frames(channel, reader, log, frame_count):
    for index in range(frame_count):
        # One credit for one frame. No RGB-D frame is loaded until requested.
        channel.sendall(b"N")
        packet = _receive(reader, log)
        if packet["index"] != index:
            raise RuntimeError(f"Expected ORB frame {index}, got {packet['index']}")
        pose = validate_pose(packet["pose"]) if packet["pose"] is not None else None
        yield OrbPose(
            index, packet["timestamp"], packet["key"], pose, packet["map_id"], packet["state"]
        )


@contextmanager
def open_orb_poses(dataset, intrinsics, depth_units_per_meter, orb_python, vocabulary, fps=None):
    """Start ORB, then yield an iterator over frame-arrival poses. Breaking stops the worker.

    Inputs use the same registered, rectified RGB-D format as offline localization.
    Loop closure is disabled for incremental fusion. Every interruption of valid
    tracking starts a new segment, including a successful relocalization.
    """
    command = [
        os.fspath(orb_python),
        "-m",
        "glassrecon.orb_stream",
        "--dataset",
        os.fspath(dataset),
        "--intrinsics",
        os.fspath(intrinsics or Path(dataset) / "intrinsics.json"),
        "--depth-units-per-meter",
        str(depth_units_per_meter),
        "--vocabulary",
        os.fspath(vocabulary),
    ]
    if fps is not None:
        command += ["--fps", str(fps)]
    channel, child = socket.socketpair()
    with channel, child, TemporaryFile() as log:
        # ORB itself writes to stdout from C++; a dedicated FD carries our protocol.
        proc = subprocess.Popen(
            command + ["--fd", str(child.fileno())],
            pass_fds=(child.fileno(),),
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
        )
        child.close()
        channel.settimeout(120)
        try:
            with channel.makefile("rb") as reader:
                ready = _receive(reader, log)
                yield _request_frames(channel, reader, log, ready["frames"])
        finally:
            channel.close()  # EOF also cancels a worker waiting for its next frame.
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        if proc.returncode:
            raise RuntimeError(f"ORB worker exited with status {proc.returncode}")


def _serve(channel, args):
    import pyorbslam3 as orb

    if not np.isfinite(args.depth_units_per_meter) or args.depth_units_per_meter <= 0:
        raise ValueError("Depth units per meter must be finite and positive")
    camera = CameraIntrinsics.load(args.intrinsics)
    frames, _ = load_sequence(args.dataset)
    fps = _camera_fps([f.timestamp for f in frames], args.fps)
    masks = args.dataset / "masks" if (args.dataset / "masks").is_dir() else None
    with TemporaryDirectory() as tmp:
        config = Path(tmp) / "settings.yaml"
        # No asynchronous loop correction or merging into an older Atlas map.
        config.write_text(_camera_settings(camera, fps) + "loopClosing: 0\n")
        slam = orb.Slam(args.vocabulary, config, orb.Sensor.RGBD, False)
        try:
            _send(channel, {"frames": len(frames)})
            map_id = -1
            previous_ok = False
            for index, frame in enumerate(frames):
                command = channel.recv(1)
                if not command:
                    return
                if command != b"N":
                    raise ValueError("Unknown ORB stream request")
                state, result = _track_raw_rgbd(
                    slam, frame, camera, args.depth_units_per_meter, masks
                )
                ok = result is not None
                if ok and not previous_ok:
                    map_id += 1
                pose = validate_pose(result.pose_wc).tolist() if ok else None
                _send(
                    channel,
                    dict(
                        index=index,
                        timestamp=frame.timestamp,
                        key=frame.key,
                        pose=pose,
                        map_id=map_id if ok else None,
                        state=state,
                    ),
                )
                previous_ok = ok
        finally:
            slam.shutdown()


def _main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", type=Path, required=True)
    cli.add_argument("--intrinsics", type=Path, required=True)
    cli.add_argument("--depth-units-per-meter", type=float, required=True)
    cli.add_argument("--vocabulary", type=Path, required=True)
    cli.add_argument("--fps", type=int)
    cli.add_argument("--fd", type=int, required=True)
    args = cli.parse_args()
    with socket.socket(fileno=args.fd) as channel:
        try:
            _serve(channel, args)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            _send(channel, {"error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    _main()
