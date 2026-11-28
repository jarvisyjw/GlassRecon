"""Online ORB tracking and bounded-latency dense mapping of a recorded RGB-D stream."""

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np
from PIL import Image

from .alignment import AlignmentConfig, AlignmentError, recover_metric_depth
from .data import CameraIntrinsics, load_sequence, matrix_quaternion
from .live_depth import DA3Worker
from .mapping import create_mapper
from .multiview import SlidingWindowRecovery
from .orb_stream import open_orb_poses
from .pipeline import prepare_output


def _write_pose(path, timestamp, pose):
    values = [timestamp, *pose[:3, 3], *matrix_quaternion(pose[:3, :3])]
    with path.open("a") as stream:
        stream.write(" ".join(f"{v:.17g}" for v in values) + "\n")


def _segment_dir(root, segment):
    path = root / f"segment_{segment:03d}"
    path.mkdir(exist_ok=True)
    return path


def _offer_latest(queue, item):
    """Single producer. Replace the oldest unstarted mapping frame when behind."""
    try:
        queue.put_nowait(item)
        return 0
    except Full:
        try:
            queue.get_nowait()
            dropped = 1
        except Empty:
            dropped = 0  # The mapper consumed it between the two calls.
        queue.put_nowait(item)
        return dropped


class _MapSegment:
    def __init__(self, segment, mapper, camera, output, viewer, refresh_every):
        self.segment = segment
        self.mapper = mapper
        self.camera = camera
        self.path = _segment_dir(output, segment)
        self.viewer = viewer
        self.refresh_every = refresh_every
        self.window = SlidingWindowRecovery(camera)
        self.first_cloud_at = None
        self.failed = 0
        (self.path / "config.json").write_text(
            json.dumps({"intrinsics_used": asdict(camera), "mapper": mapper.metadata()}, indent=2)
        )

    def integrate(self, node):
        self.mapper.integrate(node["rgb"], node["depth"], node["pose"])
        _write_pose(self.path / "fused_trajectory.txt", node["timestamp"], node["pose"])
        count = self.mapper.integrated
        if count == 1 or count % self.refresh_every == 0:
            self.publish_cloud()
        self.viewer.on_fusion(
            count,
            pending=len(self.window.pending),
            timestamp=node["timestamp"],
            stage="Mapping",
            segment=self.segment,
        )

    def publish_cloud(self):
        points, colors = self.mapper.point_cloud()
        if len(points):
            self.viewer.update_map(points, colors, segment=self.segment)
            if self.first_cloud_at is None:
                self.first_cloud_at = time.monotonic()

    def flush(self):
        for node in self.window.flush():
            self.integrate(node)

    def process(self, frame, pose, depth, units, masks, config, rng):
        rgb, sensor = frame.load(units, self.camera)
        prediction = depth.predict(rgb, frame.key)
        sensor[(sensor < config.min_depth) | (sensor > config.max_depth)] = 0
        if masks is not None:
            with Image.open(masks / f"{frame.key}.png") as image:
                valid = np.asarray(image) == 255
            if valid.shape != sensor.shape:
                raise ValueError(f"Invalid camera mask dimensions for {frame.key}")
            sensor[~valid] = 0
            rgb[~valid] = 0
            prediction = np.where(valid, prediction, 0)
        try:
            aligned, _ = recover_metric_depth(prediction, sensor, "depth", config, rng)
        except AlignmentError:
            self.failed += 1
            self.flush()
            self.window = SlidingWindowRecovery(self.camera)
            return
        node = self.window.push(frame.key, frame.timestamp, rgb, sensor, aligned, pose)
        if node is None:
            self.viewer.on_fusion(
                self.mapper.integrated,
                pending=len(self.window.pending),
                stage="Filling depth window",
                segment=self.segment,
            )
        else:
            self.integrate(node)

    def finish(self):
        self.flush()
        summary = {
            "segment": self.segment,
            "integrated": self.mapper.integrated,
            "alignment_failed": self.failed,
        }
        if self.mapper.integrated:
            self.publish_cloud()
            summary.update(self.mapper.export(self.path))
        (self.path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def _map_frames(inbox, output, camera, depth, masks, args, viewer, ready):
    segment = None
    summaries = []
    first_cloud = None
    try:
        config = AlignmentConfig(iterations=args.alignment_iterations)
        rng = np.random.default_rng(0)
        first_mapper = create_mapper(camera, backend="nvblox")
    except BaseException as exc:
        ready.set_exception(exc)
        raise
    ready.set_result(None)
    while True:
        item = inbox.get()
        if segment is not None and (item is None or item[1].map_id != segment.segment):
            summaries.append(segment.finish())
            if first_cloud is None:
                first_cloud = segment.first_cloud_at
            segment = None
        if item is None:
            return summaries, first_cloud
        frame, track = item
        if segment is None:
            mapper = (
                first_mapper
                if first_mapper is not None
                else create_mapper(camera, backend="nvblox")
            )
            first_mapper = None
            segment = _MapSegment(track.map_id, mapper, camera, output, viewer, args.viewer_every)
            del mapper
        segment.process(frame, track.pose, depth, args.depth_units_per_meter, masks, config, rng)


def run_live(args, viewer):
    dataset = Path(args.dataset)
    calibration = Path(args.intrinsics or dataset / "intrinsics.json")
    camera = CameraIntrinsics.load(calibration)
    if not Path(args.orb_vocabulary).is_file():
        raise FileNotFoundError(f"Missing ORB-SLAM3 vocabulary: {args.orb_vocabulary}")
    frames, _ = load_sequence(dataset)
    frames = frames[: args.max_frames]
    output = prepare_output(args.output)
    masks = dataset / "masks" if (dataset / "masks").is_dir() else None
    viewer.configure(camera, len(frames), "ORB-SLAM3", "GlassRecon + nvblox")
    if not args.autoplay:
        viewer.wait_for_start()
    viewer.on_fusion(0, stage="Loading ORB-SLAM3, DA3 and nvblox")
    tracked = dropped = processed = 0
    segment_counts = {}
    first_pose = None
    interrupted = False
    last_segment = 0
    # The tracking thread only hands off file references and online poses. The GPU
    # mapper owns its TSDF and five-frame window; it never waits for a future pose file.
    inbox = Queue(maxsize=2)
    with (
        open_orb_poses(
            dataset,
            calibration,
            args.depth_units_per_meter,
            args.orb_python,
            args.orb_vocabulary,
            fps=args.fps,
        ) as poses,
        DA3Worker(args.model_python, model_id=args.model_id) as depth,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvblox") as pool,
    ):
        ready = Future()
        worker = pool.submit(_map_frames, inbox, output, camera, depth, masks, args, viewer, ready)
        try:
            ready.result()
        except BaseException:
            inbox.put_nowait(None)
            raise
        start = time.monotonic()
        try:
            for index, frame in enumerate(frames):
                deadline = start + (frame.timestamp - frames[0].timestamp) / args.playback_rate
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if worker.done():
                    worker.result()
                track = next(poses, None)
                if (
                    track is None
                    or track.index != index
                    or track.key != frame.key
                    or track.timestamp != frame.timestamp
                ):
                    raise RuntimeError(f"ORB stream does not match source frame {frame.key}")
                if track.pose is not None:
                    last_segment = track.map_id
                processed += 1
                viewer.on_tracking(
                    processed, frame.timestamp, track.state, track.pose, segment=last_segment
                )
                if track.pose is None:
                    continue
                if first_pose is None:
                    first_pose = time.monotonic() - start
                tracked += 1
                segment_counts[last_segment] = segment_counts.get(last_segment, 0) + 1
                _write_pose(
                    _segment_dir(output, last_segment) / "trajectory.txt",
                    frame.timestamp,
                    track.pose,
                )
                dropped += _offer_latest(inbox, (frame, track))
        except KeyboardInterrupt:
            interrupted = True
        finally:
            tracking_seconds = time.monotonic() - start
            # Preserve the pending frames on EOF. Also escape a full queue if
            # the mapper failed, so its original exception reaches the caller.
            while not worker.done():
                try:
                    inbox.put(None, timeout=0.1)
                    break
                except Full:
                    pass
        segments, cloud_at = worker.result()
        elapsed = time.monotonic() - start
    fused = sum(s["integrated"] for s in segments)
    mapped = {s["segment"]: s for s in segments}
    for segment, count in segment_counts.items():
        info = mapped.setdefault(
            segment, {"segment": segment, "integrated": 0, "alignment_failed": 0}
        )
        info["tracked"] = count
    summary = {
        "selected": len(frames),
        "processed": processed,
        "tracked": tracked,
        "integrated": fused,
        "tracking_unavailable": processed - tracked,
        "mapping_queue_dropped": dropped,
        "alignment_failed": sum(s["alignment_failed"] for s in segments),
        "tracking_hz": processed / tracking_seconds,
        "fusion_hz": fused / elapsed,
        "first_pose_seconds": first_pose,
        "first_cloud_seconds": cloud_at - start if cloud_at else None,
        "elapsed_seconds": elapsed,
        "playback_rate": args.playback_rate,
        "alignment_iterations": args.alignment_iterations,
        "interrupted": interrupted,
        "segments": [mapped[key] for key in sorted(mapped)],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not fused and not interrupted:
        raise RuntimeError(f"No frames could be fused. See {output / 'summary.json'}")
    viewer.on_fusion(
        mapped.get(last_segment, {}).get("integrated", 0),
        stage="Replay complete",
        segment=last_segment,
    )
    return summary


def serve_live(args, viewer):
    """Keep the finished map visible and let the browser start another live run."""
    run = 1
    current = copy(args)
    while True:
        summary = run_live(current, viewer)
        print(json.dumps(summary, indent=2), flush=True)
        if args.exit_after_run or summary["interrupted"]:
            return
        viewer.wait_for_replay()
        run += 1
        current.autoplay = True
        current.output = str(Path(args.output) / "replays" / f"run_{run:03d}")
