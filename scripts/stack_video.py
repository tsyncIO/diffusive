#!/usr/bin/env python3
"""
Static-Path Batch Video Stacking Runner (No-Trimming, Full Verification).

This script scans a directory for JSON metadata files, locates corresponding
real (input) and fake (generated) videos, and stacks them side-by-side with
prompt overlays, video statistics, and resolution information.

Usage:
    python3 stack_video.py --json-dir /path/to/jsons --video-dir /path/to/reals --output-dir /path/to/stacks
"""

import os
import sys
import json
import random
import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Generator, Optional, Dict, List, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


@dataclass
class InferenceJob:
    """Container for a single video pair to stack."""
    data_id: str
    input_video: Path          # real video
    output_video: Path         # fake/generated video
    diff_video: Optional[Path] # difference video
    prompt: str
    metadata: Dict


class SchemaParser:
    """Handles directory scanning for jobs matching inference_batch.py logic."""
    
    def __init__(self, json_dir: Path, video_dir: Path, verbose: bool = False):
        self.json_dir = json_dir
        self.video_dir = video_dir
        self.verbose = verbose

    def scan(self, json_files: Optional[List[Path]] = None) -> Generator[InferenceJob, None, None]:
        """Iterates through JSON files in the json_dir."""
        logger.info("Starting job discovery for stacking...")
        logger.info(f"JSON Directory:  {self.json_dir}")
        logger.info(f"Video Directory: {self.video_dir}")
        
        if not self.json_dir.exists():
            logger.error(f"JSON directory does not exist: {self.json_dir}")
            return

        if not self.video_dir.exists():
            logger.error(f"Video directory does not exist: {self.video_dir}")
            return

        if json_files is None:
            json_files = sorted([f for f in self.json_dir.glob("*.json") if f.name != ".checkpoint_batch.json"])
        
        found_count = 0
        for json_file in json_files:
            if self.verbose:
                logger.info(f"Processing JSON: {json_file.name}")
            
            job = self._parse_json(json_file)
            if job:
                found_count += 1
                yield job
        
        logger.info(f"Discovery completed. Total valid pairs found: {found_count}")

    def _parse_json(self, json_file: Path) -> Optional[InferenceJob]:
        """Parses a single JSON file and matches it to its real/fake videos."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            data_id = data.get("data_id")
            target_data = data.get("target", {})
            target_id = target_data.get("data_id")
            prompt = data.get("gen_prompt", "")

            if not all([data_id, target_id]):
                logger.warning(f"Skipping {json_file.name}: Missing data_id or target.data_id")
                return None

            input_video = self.video_dir / f"{target_id}.mp4"
            output_video = json_file.parent / f"{data_id}.mp4"
            
            # Diff video is in a subfolder: diff_{data_id}/diff_{data_id}.mp4
            diff_video = json_file.parent / f"diff_{data_id}" / f"diff_{data_id}.mp4"
            if not diff_video.exists():
                diff_video = None

            if self.verbose:
                logger.info(f"--- Mapping Pair: {data_id} ---")
                logger.info(f"  JSON Source:  {json_file}")
                logger.info(f"  Prompt:       {prompt[:100]}...")
                logger.info(f"  Real Video:   {input_video}")
                logger.info(f"  Fake Video:   {output_video}")
                logger.info("-" * (len(data_id) + 20))

            if not output_video.exists():
                if self.verbose:
                    logger.debug(f"Fake video missing for {data_id} (inference likely pending): {output_video}")
                return None

            if not input_video.exists():
                logger.error(f"Real video missing for {target_id}: {input_video}")
                return None
            
            return InferenceJob(
                data_id=data_id,
                input_video=input_video,
                output_video=output_video,
                diff_video=diff_video,
                prompt=prompt,
                metadata=data
            )
        except Exception as e:
            logger.error(f"Failed to parse {json_file.name}: {e}")
            return None


class CheckpointManager:
    """Manages checkpoint file for stacking progress."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_file = checkpoint_dir / ".checkpoint_stack.json"
        self.completed_ids: Dict[str, dict] = self._load()

    def _load(self) -> Dict[str, dict]:
        if not self.checkpoint_file.exists():
            return {}
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return {}

    def is_completed(self, data_id: str) -> bool:
        return data_id in self.completed_ids

    def mark_completed(self, job: InferenceJob):
        self.completed_ids[job.data_id] = {
            "timestamp": datetime.now().isoformat(),
            "status": "stacked",
            "real": str(job.input_video),
            "fake": str(job.output_video)
        }
        self._save()

    def _save(self):
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.completed_ids, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")


class VideoStacker:
    """Handles video stacking with frame‑by‑frame verification, stats overlay, and resolution display."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug

    def process(self, job: InferenceJob, output_path: Path) -> bool:
        """Stack real, fake, and diff videos side‑by‑side without trimming, showing stats and resolutions."""
        if self.debug:
            logger.info(f"Stacking: {job.data_id}")

        # Read videos and their properties
        frames_real, props_real = self._read_video_with_props(job.input_video)
        if not frames_real:
            logger.error(f"Real video empty or unreadable: {job.input_video}")
            return False

        frames_fake, props_fake = self._read_video_with_props(job.output_video)
        if not frames_fake:
            logger.warning(f"Fake video empty: {job.output_video}, using black frames")
            frames_fake = [np.zeros_like(frames_real[0]) for _ in range(len(frames_real))]
            props_fake = {
                'fps': props_real['fps'],
                'frame_count': 0,
                'duration': 0.0,
                'width': props_real['width'],
                'height': props_real['height']
            }

        # ---- RESOLUTION DISPLAY (CONSOLE) ----
        res_real = f"{props_real['width']}x{props_real['height']}"
        res_fake = f"{props_fake['width']}x{props_fake['height']}"
        logger.info("┌─────────────────────────────────────────┐")
        logger.info(f"│ INPUT  resolution: {res_real:23} │")
        logger.info(f"│ OUTPUT resolution: {res_fake:23} │")
        if props_real['width'] == props_fake['width'] and props_real['height'] == props_fake['height']:
            logger.info("│ ✓ RESOLUTIONS MATCH                          │")
        else:
            logger.info("│ ✗ RESOLUTIONS DIFFER                         │")
        logger.info("└─────────────────────────────────────────┘")
        logger.info(f"Real video: {res_real} @ {props_real['fps']:.2f} fps, {props_real['frame_count']} frames")
        logger.info(f"Fake video: {res_fake} @ {props_fake['fps']:.2f} fps, {props_fake['frame_count']} frames")

        # Resize fake frames to match real dimensions (if needed)
        target_shape = frames_real[0].shape[:2]  # (height, width)
        if frames_fake[0].shape[:2] != target_shape:
            logger.info(f"Resizing fake video from {frames_fake[0].shape[1]}x{frames_fake[0].shape[0]} to {target_shape[1]}x{target_shape[0]}")
            frames_fake = self._resize_frames(frames_fake, target_shape)

        # Read diff video if it exists
        frames_diff = []
        props_diff = {}
        if job.diff_video:
            frames_diff, props_diff = self._read_video_with_props(job.diff_video)
            if frames_diff:
                if frames_diff[0].shape[:2] != target_shape:
                    logger.info(f"Resizing diff video from {frames_diff[0].shape[1]}x{frames_diff[0].shape[0]} to {target_shape[1]}x{target_shape[0]}")
                    frames_diff = self._resize_frames(frames_diff, target_shape)
            else:
                logger.warning(f"Diff video present but unreadable: {job.diff_video}")

        if not frames_diff:
            # Fallback to black frames if missing
            frames_diff = [np.zeros_like(frames_real[0]) for _ in range(len(frames_real))]
            props_diff = {
                'fps': props_real['fps'],
                'frame_count': 0,
                'duration': 0.0,
                'width': props_real['width'],
                'height': props_real['height']
            }

        # Compare frame counts
        len_real = len(frames_real)
        len_fake = len(frames_fake)
        mismatch = (len_real != len_fake)
        if mismatch:
            logger.warning(
                f"Frame count mismatch for {job.data_id}: "
                f"Real={len_real}, Fake={len_fake}"
            )

        # Banner heights (increased to accommodate resolution line)
        banner_top_h = 110    # extra space for resolution display
        banner_bottom_h = 50

        # Prepare per‑frame statistics strings
        stats_text_real = self._format_stats("Real", props_real, len_real)
        stats_text_fake = self._format_stats("Fake", props_fake, len_fake)
        stats_text_diff = self._format_stats("Diff", props_diff, len(frames_diff))
        mismatch_text = "⚠ FRAME COUNT MISMATCH ⚠" if (len_real != len_fake or len_real != len(frames_diff)) else ""

        output_frames = []
        max_frames = max(len_real, len_fake, len(frames_diff))

        for i in range(max_frames):
            f_real = frames_real[i] if i < len_real else self._blank_frame(target_shape, "Real frame missing")
            f_fake = frames_fake[i] if i < len_fake else self._blank_frame(target_shape, "Fake frame missing")
            f_diff = frames_diff[i] if i < len(frames_diff) else self._blank_frame(target_shape, "Diff frame missing")
            
            stacked = self._create_stacked_frame(
                f_real, f_fake, f_diff,
                job.prompt,
                job.metadata.get("target", {}).get("data_id", ""),
                job.data_id,
                stats_text_real, stats_text_fake, stats_text_diff, mismatch_text,
                banner_top_h, banner_bottom_h,
                res_real, res_fake, f"{props_diff['width']}x{props_diff['height']}"
            )
            output_frames.append(stacked)

        # Write output video (using real video's FPS, or fallback to 25)
        output_fps = props_real['fps'] if props_real['fps'] > 0 else 25.0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = output_frames[0].shape[:2]
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (w, h))
        for frame in output_frames:
            out.write(frame)
        out.release()
        
        logger.info(f"Stacked video output: {w}x{h} @ {output_fps:.2f} fps, {len(output_frames)} frames -> {output_path}")
        return True

    @staticmethod
    def _read_video_with_props(video_path: Path) -> Tuple[List[np.ndarray], Dict]:
        """Read all frames and return metadata (fps, frame count, duration, width, height)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        props = {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height
        }
        return frames, props

    @staticmethod
    def _resize_frames(frames: List[np.ndarray], target_shape) -> List[np.ndarray]:
        h, w = target_shape
        return [cv2.resize(f, (w, h)) for f in frames]

    @staticmethod
    def _format_stats(label: str, props: Dict, actual_frames: int) -> str:
        """Create a compact stats string including resolution."""
        return (f"{label}: {props['width']}x{props['height']}, "
                f"{props['frame_count']} frm, {props['fps']:.2f} fps, "
                f"{props['duration']:.2f}s (read: {actual_frames})")

    @staticmethod
    def _blank_frame(shape, text: str) -> np.ndarray:
        """Create a black frame with an error message."""
        h, w = shape
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(frame, text, (x, y), font, font_scale,
                    (0, 0, 255), thickness, cv2.LINE_AA)
        return frame

    def _create_stacked_frame(self, f_real: np.ndarray, f_fake: np.ndarray, f_diff: np.ndarray,
                              prompt: str, real_id: str, fake_id: str,
                              stats_real: str, stats_fake: str, stats_diff: str, mismatch_text: str,
                              top_h: int, bottom_h: int,
                              res_real: str, res_fake: str, res_diff: str) -> np.ndarray:
        """Build final stacked frame with banners, labels, statistics, and resolution overlay."""
        # Horizontal stack (1x3)
        combined = np.hstack([f_real, f_fake, f_diff])
        H, W = combined.shape[:2]
        one_third = W // 3
        two_thirds = (2 * W) // 3

        # ---- TOP BANNER (Prompt + Stats + Resolution) ----
        top_banner = np.zeros((top_h, W, 3), dtype=np.uint8)
        y_pos = 20
        # Prompt line
        top_banner = self._draw_text_centered(top_banner, prompt, y=y_pos, font_scale=0.55)
        y_pos += 30
        # Stats line
        top_banner = self._draw_text_centered_in_region(top_banner, stats_real, 0, one_third, y=y_pos, font_scale=0.45)
        top_banner = self._draw_text_centered_in_region(top_banner, stats_fake, one_third, two_thirds, y=y_pos, font_scale=0.45)
        top_banner = self._draw_text_centered_in_region(top_banner, stats_diff, two_thirds, W, y=y_pos, font_scale=0.45)
        y_pos += 25
        # Resolution line
        top_banner = self._draw_text_centered_in_region(top_banner, f"In: {res_real}", 0, one_third, y=y_pos, font_scale=0.5)
        top_banner = self._draw_text_centered_in_region(top_banner, f"Out: {res_fake}", one_third, two_thirds, y=y_pos, font_scale=0.5)
        top_banner = self._draw_text_centered_in_region(top_banner, f"Diff: {res_diff}", two_thirds, W, y=y_pos, font_scale=0.5)
        y_pos += 25
        # Mismatch warning if any
        if mismatch_text:
            top_banner = self._draw_text_centered(top_banner, mismatch_text, y=y_pos, color=(0, 0, 255), font_scale=0.6)

        # ---- SIDE LABELS (Original / Edited / Difference) ----
        combined = self._draw_text_centered_in_region(combined, "Original", 0, one_third, y=30, font_scale=0.7)
        combined = self._draw_text_centered_in_region(combined, "Edited", one_third, two_thirds, y=30, font_scale=0.7)
        combined = self._draw_text_centered_in_region(combined, "Difference", two_thirds, W, y=30, font_scale=0.7)

        # ---- BOTTOM BANNER (Video IDs) ----
        bottom_banner = np.zeros((bottom_h, W, 3), dtype=np.uint8)
        bottom_banner = self._draw_text_centered_in_region(bottom_banner, f"Real ID: {real_id}", 0, one_third, y=25, font_scale=0.5)
        bottom_banner = self._draw_text_centered_in_region(bottom_banner, f"Fake ID: {fake_id}", one_third, two_thirds, y=25, font_scale=0.5)
        bottom_banner = self._draw_text_centered_in_region(bottom_banner, f"Diff ID: {fake_id}", two_thirds, W, y=25, font_scale=0.5)

        # Vertical stack
        final = np.vstack([top_banner, combined, bottom_banner])
        return final

    @staticmethod
    def _draw_text_centered(img: np.ndarray, text: str, y: int, color=(255, 255, 255), font_scale: float = 0.6) -> np.ndarray:
        """Draw multi-line text centered horizontally on the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        h, w = img.shape[:2]
        max_width = w - 40

        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test = f"{current_line} {word}" if current_line else word
            (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
            if tw > max_width:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test
        if current_line:
            lines.append(current_line)

        y_pos = y
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            x = (w - tw) // 2
            cv2.putText(img, line, (x, y_pos), font, font_scale,
                        color, thickness, cv2.LINE_AA)
            y_pos += th + 6
        return img

    @staticmethod
    def _draw_text_centered_in_region(img: np.ndarray, text: str,
                                      x_start: int, x_end: int, y: int,
                                      color=(255, 255, 255), font_scale: float = 0.6) -> np.ndarray:
        """Draw text centered within a horizontal region."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        region_width = x_end - x_start
        max_width = region_width - 20

        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test = f"{current_line} {word}" if current_line else word
            (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
            if tw > max_width:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test
        if current_line:
            lines.append(current_line)

        y_pos = y
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            x = x_start + (region_width - tw) // 2
            cv2.putText(img, line, (x, y_pos), font, font_scale,
                        color, thickness, cv2.LINE_AA)
            y_pos += th + 6
        return img


class BatchStackingRunner:
    """Manages the execution loop and checkpointing for stacking."""
    
    def __init__(self, output_dir: Path, skip_completed: bool = True, debug: bool = False):
        self.output_dir = output_dir
        self.skip_completed = skip_completed
        self.debug = debug
        self.checkpoint_mgr = CheckpointManager(output_dir)
        self.stacker = VideoStacker(debug=debug)

    def run(self, jobs: List[InferenceJob]):
        if not jobs:
            logger.info("No jobs to process.")
            return

        success_count = 0
        fail_count = 0
        skipped_count = 0

        logger.info(f"Starting batch stacking for {len(jobs)} jobs...")
        start_time = time.time()

        for job in tqdm(jobs, desc="Stacking Videos"):
            if self.skip_completed and self.checkpoint_mgr.is_completed(job.data_id):
                skipped_count += 1
                continue

            output_path = self.output_dir / f"{job.data_id}_stacked.mp4"
            logger.info(f"Processing: {job.data_id}")

            if self.stacker.process(job, output_path):
                success_count += 1
                self.checkpoint_mgr.mark_completed(job)
            else:
                fail_count += 1
                logger.error(f"Stacking FAILED: {job.data_id}")

        end_time = time.time()
        elapsed = end_time - start_time
        avg_time = elapsed / max(1, (success_count + fail_count))

        logger.info("\n" + "="*40)
        logger.info("BATCH STACKING SUMMARY")
        logger.info(f"  Total Jobs Found: {len(jobs)}")
        logger.info(f"  Successful:       {success_count}")
        logger.info(f"  Failed:           {fail_count}")
        logger.info(f"  Skipped (Comp):   {skipped_count}")
        logger.info(f"  Total Time:       {elapsed:.2f}s")
        if (success_count + fail_count) > 0:
            logger.info(f"  Avg Time/Job:     {avg_time:.2f}s")
        logger.info("="*40)


def main():
    parser = argparse.ArgumentParser(description="Stack Real and Fake videos side-by-side with verification and resolution display.")
    parser.add_argument("--json-dir", type=str, required=True,
                        help="Directory containing metadata JSONs and fake videos")
    parser.add_argument("--video-dir", type=str, required=True,
                        help="Directory containing original real videos")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for stacked output videos")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of pairs to process (arbitrary order)")
    parser.add_argument("--latest", type=int, default=None,
                        help="Fetch and process only the N most recently generated fake videos")
    parser.add_argument("--random", type=int, default=None,
                        help="Fetch and process N random fake videos")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                        help="Skip already stacked videos (uses checkpoint file)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print mapping without processing")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed path mappings during discovery")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    json_dir = Path(args.json_dir).resolve()
    video_dir = Path(args.video_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    schema_parser = SchemaParser(json_dir=json_dir, video_dir=video_dir, verbose=args.verbose)
    
    if args.latest or args.random or args.limit:
        logger.info("Fast discovering completed files using os.scandir...")
        mp4_entries = []
        try:
            for entry in os.scandir(json_dir):
                if entry.is_file() and entry.name.endswith(".mp4") and not entry.name.startswith("diff_"):
                    mp4_entries.append(entry)
        except Exception as e:
            logger.error(f"Failed to scan directory: {e}")
            return
            
        if args.latest:
            logger.info(f"Sorting {len(mp4_entries)} completed videos by modification time...")
            mp4_entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
            mp4_entries = mp4_entries[:args.latest * 3]
        elif args.random:
            random.shuffle(mp4_entries)
            mp4_entries = mp4_entries[:args.random * 3]
        elif args.limit:
            mp4_entries = mp4_entries[:args.limit * 3]
            
        json_files = []
        for e in mp4_entries:
            json_path = json_dir / f"{Path(e.path).stem}.json"
            if json_path.exists():
                json_files.append(json_path)
                
        all_jobs = list(schema_parser.scan(json_files))
        
        if args.latest:
            all_jobs.sort(key=lambda job: job.output_video.stat().st_mtime, reverse=True)
            all_jobs = all_jobs[:args.latest]
        elif args.random:
            random.shuffle(all_jobs)
            all_jobs = all_jobs[:args.random]
        elif args.limit:
            all_jobs = all_jobs[:args.limit]
    else:
        # Fallback to normal scan
        all_jobs = list(schema_parser.scan())

    if args.dry_run:
        logger.info("\n" + "="*40)
        logger.info("DRY RUN: DISCOVERED PAIRS")
        for job in all_jobs:
            # Get resolutions for display
            cap = cv2.VideoCapture(str(job.input_video))
            if cap.isOpened():
                real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            else:
                real_w = real_h = "unknown"
            cap = cv2.VideoCapture(str(job.output_video))
            if cap.isOpened():
                fake_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fake_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            else:
                fake_w = fake_h = "unknown"
            logger.info(f"ID:     {job.data_id}")
            logger.info(f"Real:   {job.input_video} ({real_w}x{real_h})")
            logger.info(f"Fake:   {job.output_video} ({fake_w}x{fake_h})")
            logger.info(f"Diff:   {job.diff_video if job.diff_video else '[MISSING]'}")
            logger.info(f"Prompt: {job.prompt[:100]}...")
            logger.info("-" * 20)
        logger.info("="*40)
        return

    if not all_jobs:
        logger.info("No valid video pairs found to stack.")
        return

    runner = BatchStackingRunner(
        output_dir=output_dir,
        skip_completed=args.skip_completed,
        debug=args.debug
    )
    runner.run(all_jobs)


if __name__ == "__main__":
    main()