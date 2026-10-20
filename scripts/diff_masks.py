

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _extract_frame_number(path: Path, prefix: str) -> int:
    match = re.match(rf"^{re.escape(prefix)}(\d+)\.png$", path.name)
    if match:
        return int(match.group(1))
    return -1


def diff_masks(input_frame, output_frame, frame_number, json_path, mp4_duration, mp4_fps, last_frame=False):
    input_arr = np.asarray(input_frame)
    output_arr = np.asarray(output_frame)

    if input_arr.shape != output_arr.shape:
        raise ValueError(
            f"input_frame and output_frame must have identical shape, got "
            f"{input_arr.shape} vs {output_arr.shape}"
        )

    if input_arr.ndim != 3 or 3 not in input_arr.shape:
        raise ValueError(
            "Expected RGB arrays with 3 channels (shape like HxWx3 or 3xHxW)."
        )

    if input_arr.shape[-1] != 3 and input_arr.shape[0] == 3:
        input_arr = np.moveaxis(input_arr, 0, -1)
        output_arr = np.moveaxis(output_arr, 0, -1)

    if mp4_duration <= 0:
        raise ValueError(f"mp4_duration must be > 0, got {mp4_duration}")

    if mp4_fps <= 0:
        raise ValueError(f"mp4_fps must be > 0, got {mp4_fps}")

    # Maps [-255, 255] differences to [0, 255].
    diff = np.rint(((input_arr.astype(np.int16) - output_arr.astype(np.int16)) + 255) / 2.0)
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    mp4_file = Path(json_path)
    mp4_basename = mp4_file.stem
    out_dir = mp4_file.parent / f"diff_{mp4_basename}"
    out_dir.mkdir(parents=True, exist_ok=True)

    png_name = f"diff_{mp4_basename}_{frame_number}.png"
    png_path = out_dir / png_name
    Image.fromarray(diff, mode="RGB").save(png_path)

    if not last_frame:
        return str(png_path)

    prefix = f"diff_{mp4_basename}_"
    png_files = [p for p in out_dir.glob(f"{prefix}*.png") if _extract_frame_number(p, prefix) >= 0]
    png_files.sort(key=lambda p: _extract_frame_number(p, prefix))

    if not png_files:
        raise RuntimeError(f"No diff PNG files found in {out_dir}")

    expected_frames = int(round(float(mp4_duration) * float(mp4_fps)))
    if expected_frames <= 0:
        raise ValueError(
            f"Invalid expected frame count from mp4_duration={mp4_duration} and mp4_fps={mp4_fps}."
        )

    # Enforce output duration and pace by making frame count exactly duration * fps.
    if len(png_files) < expected_frames:
        png_files.extend([png_files[-1]] * (expected_frames - len(png_files)))
    elif len(png_files) > expected_frames:
        png_files = png_files[:expected_frames]

    out_mp4 = out_dir / f"diff_{mp4_basename}.mp4"
    cmd = None
    try:
        with tempfile.TemporaryDirectory(dir=out_dir) as temp_dir:
            temp_dir_path = Path(temp_dir)
            for index, source_path in enumerate(png_files):
                sequence_path = temp_dir_path / f"frame_{index:06d}.png"
                try:
                    sequence_path.symlink_to(source_path)
                except OSError:
                    sequence_path.write_bytes(source_path.read_bytes())

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(mp4_fps),
                "-i",
                str(temp_dir_path / "frame_%06d.png"),
                "-frames:v",
                str(expected_frames),
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "veryslow",
                "-pix_fmt",
                "rgb24",
                str(out_mp4),
            ]

            subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to build the output mp4 but was not found in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed while creating {out_mp4}: {exc.stderr}") from exc

    return str(png_path), str(out_mp4)