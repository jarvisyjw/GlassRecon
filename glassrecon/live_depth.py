"""Run the existing DA3 predictor in a resident, separately configured Python process."""

import argparse
import json
import os
import socket
import subprocess
from multiprocessing.connection import Connection

import numpy as np

from .predictors import create_predictor


class DA3Worker:
    """One synchronous relative-depth predictor for an ordered stream of RGB frames."""

    representation = "depth"

    def __init__(self, model_python, model_id=None, device="cuda", process_res=504):
        self.model_id = str(model_id or "depth-anything/DA3-LARGE")
        if process_res <= 0:
            raise ValueError("DA3 process resolution must be positive")
        self._command = [
            str(model_python),
            "-m",
            "glassrecon.live_depth",
            "--model-id",
            self.model_id,
            "--device",
            device,
            "--process-res",
            str(process_res),
        ]
        self._connection = None
        self._process = None

    def __enter__(self):
        if self._process is not None:
            raise RuntimeError("DA3 worker is already running")
        # Keep native library output and download progress off the binary data channel.
        local, remote = socket.socketpair()
        with local, remote:
            self._process = subprocess.Popen(
                [*self._command, "--fd", str(remote.fileno())], pass_fds=(remote.fileno(),)
            )
            self._connection = Connection(local.detach())
        try:
            self._check_status("initializing the model")
        except BaseException:
            self.close()
            raise
        return self

    def _check_status(self, operation):
        try:
            message = self._connection.recv_bytes()
        except (OSError, EOFError) as exc:
            raise RuntimeError(f"Lost the DA3 worker while {operation}") from exc
        if message != b"OK":
            raise RuntimeError(f"DA3 failed while {operation}: {message.decode(errors='replace')}")

    def predict(self, rgb, key):
        if self._connection is None:
            raise RuntimeError("Start the DA3 worker before predicting")
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
            raise ValueError("DA3 expects a nonempty H x W x 3 uint8 RGB frame")
        h, w = rgb.shape[:2]
        try:
            self._connection.send_bytes(json.dumps([str(key), h, w]).encode())
            self._connection.send_bytes(memoryview(np.ascontiguousarray(rgb)).cast("B"))
            self._check_status(f"predicting {key}")
            data = self._connection.recv_bytes(maxlength=h * w * 4)
        except (OSError, EOFError) as exc:
            raise RuntimeError(f"Lost the DA3 worker while transferring {key}") from exc
        if len(data) != h * w * 4:
            raise RuntimeError(f"Incomplete DA3 depth map for {key}")
        return np.frombuffer(data, dtype="<f4").reshape(h, w).copy()

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            finally:
                self._process = None

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _run(connection, model_id, device, process_res):
    predictor = create_predictor("da3", model_id, device, process_res=process_res)
    connection.send_bytes(b"OK")
    while True:
        try:
            request = connection.recv_bytes()
        except EOFError:
            return
        key, h, w = json.loads(request)
        pixels = connection.recv_bytes(maxlength=h * w * 3)
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 3)
        depth = np.asarray(predictor.predict(rgb, key), dtype="<f4", order="C")
        if depth.shape != (h, w):
            raise ValueError(f"DA3 returned depth dimensions {depth.shape} for {key}")
        connection.send_bytes(b"OK")
        connection.send_bytes(memoryview(depth).cast("B"))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fd", type=int, required=True)
    cli.add_argument("--model-id", required=True)
    cli.add_argument("--device", required=True)
    cli.add_argument("--process-res", type=int, required=True)
    args = cli.parse_args()
    os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
    with Connection(args.fd) as connection:
        try:
            _run(connection, args.model_id, args.device, args.process_res)
        except Exception as exc:
            try:
                connection.send_bytes(f"{type(exc).__name__}: {exc}".encode())
            except OSError:
                pass  # Parent already closed its channel.
            raise


if __name__ == "__main__":
    main()
