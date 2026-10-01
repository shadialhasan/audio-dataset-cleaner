# -*- coding: utf-8 -*-
"""
Audio Dataset Cleaner & Preprocessor for Voice Cloning (XTTS-v2 / RVC / VITS)
Author: Eng. MHD. Shadi AL-Hasan <mhd.shadi.alhasan@gmail.com>
Phone: +963934005922
Copyright (c) 2026 MHD. Shadi AL-Hasan
"""

import os
import sys
import csv
import json
import wave
import struct
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Enable VT100 colors on legacy Windows consoles if possible
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


class Logger:
    """Terminal logger with ANSI colors and cross-platform fallback."""
    COLORS = {
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
        "CYAN": "\033[36m",
        "GREEN": "\033[32m",
        "YELLOW": "\033[33m",
        "RED": "\033[31m",
        "MAGENTA": "\033[35m",
        "GRAY": "\033[90m",
    }

    def __init__(self, color_enabled: bool = True):
        self.color_enabled = color_enabled and (
            hasattr(sys.stdout, "isatty") and sys.stdout.isatty() or os.getenv("FORCE_COLOR") == "1"
        )

    def _c(self, color_name: str, text: str) -> str:
        if self.color_enabled and color_name in self.COLORS:
            return f"{self.COLORS[color_name]}{text}{self.COLORS['RESET']}"
        return text

    def info(self, msg: str):
        print(f"{self._c('CYAN', '[*]')} {msg}")

    def success(self, msg: str):
        print(f"{self._c('GREEN', '[+]')} {msg}")

    def warning(self, msg: str):
        print(f"{self._c('YELLOW', '[!]')} {msg}")

    def error(self, msg: str):
        print(f"{self._c('RED', '[-] ERROR:')} {msg}", file=sys.stderr)

    def skip(self, msg: str):
        print(f"{self._c('GRAY', '[SKIP]')} {msg}")


log = Logger()


def load_env_file(filepath: Path) -> Dict[str, str]:
    """Parse key=value pairs from a .env file without external dependencies."""
    env_vars: Dict[str, str] = {}
    if not filepath.exists() or not filepath.is_file():
        return env_vars
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                env_vars[key.strip()] = val.strip().strip('"').strip("'")
    except Exception as exc:
        log.warning(f"Could not parse environment file {filepath}: {exc}")
    return env_vars


def load_config_file(filepath: Path) -> Dict[str, Any]:
    """Load configuration dictionary from JSON file."""
    if not filepath.exists() or not filepath.is_file():
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning(f"Could not load config file {filepath}: {exc}")
        return {}


class AudioDatasetCleaner:
    """Preprocesses audio datasets for TTS and voice cloning pipelines."""

    def __init__(
        self,
        target_sr: int = 22050,
        silence_thresh_db: float = -40.0,
        min_duration: float = 1.0,
        max_duration: float = 12.0,
        norm_peak_ratio: float = 0.9,
        window_size_ms: int = 20,
        logger: Optional[Logger] = None,
    ):
        if min_duration <= 0:
            raise ValueError(f"min_duration must be positive, got {min_duration}")
        if max_duration < min_duration:
            raise ValueError(f"max_duration ({max_duration}) cannot be less than min_duration ({min_duration})")
        if target_sr <= 0:
            raise ValueError(f"target_sr must be positive, got {target_sr}")

        self.target_sr = int(target_sr)
        self.silence_thresh_db = float(silence_thresh_db)
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)
        self.norm_peak_ratio = float(norm_peak_ratio)
        self.window_size_ms = int(window_size_ms)
        self.log = logger or log

    @staticmethod
    def calculate_rms(samples: List[int]) -> float:
        """Calculate Root Mean Square (RMS) energy for sample list."""
        if not samples:
            return 0.0
        sum_squares = sum(s * s for s in samples)
        return math.sqrt(sum_squares / len(samples))

    @staticmethod
    def db_to_linear(db: float) -> float:
        """Convert decibel value to linear amplitude factor."""
        if db == -float("inf"):
            return 0.0
        return 10.0 ** (db / 20.0)

    def process_wav(self, input_path: Path, output_path: Path) -> Tuple[bool, float, str]:
        """
        Processes a single WAV file:
        - Validates 16-bit PCM format
        - Converts stereo to mono
        - Checks raw duration bounds
        - Trims leading and trailing silence
        - Normalizes peak amplitude
        - Writes normalized mono WAV

        Returns: (success: bool, duration: float, message: str)
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            msg = f"File does not exist: {input_path}"
            self.log.error(msg)
            return False, 0.0, msg

        try:
            with wave.open(str(input_path), "rb") as wf:
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                num_frames = wf.getnframes()
                raw_data = wf.readframes(num_frames)
        except wave.Error as exc:
            msg = f"Invalid WAV header or unsupported format: {exc}"
            self.log.skip(f"{input_path.name}: {msg}")
            return False, 0.0, msg
        except Exception as exc:
            msg = f"Failed opening file: {exc}"
            self.log.error(f"{input_path.name}: {msg}")
            return False, 0.0, msg

        if sample_width != 2:
            msg = f"Only 16-bit PCM supported directly (sample_width={sample_width})"
            self.log.skip(f"{input_path.name}: {msg}")
            return False, 0.0, msg

        if num_frames == 0 or len(raw_data) == 0:
            msg = "WAV file contains no audio frames"
            self.log.skip(f"{input_path.name}: {msg}")
            return False, 0.0, msg

        try:
            fmt = f"<{num_frames * num_channels}h"
            samples = list(struct.unpack(fmt, raw_data))
        except struct.error as exc:
            msg = f"Failed unpacking audio PCM frames: {exc}"
            self.log.error(f"{input_path.name}: {msg}")
            return False, 0.0, msg

        # Convert stereo to mono by averaging channels
        if num_channels == 2:
            mono_samples = [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)]
        elif num_channels == 1:
            mono_samples = samples
        else:
            msg = f"Multi-channel audio ({num_channels} ch) not supported"
            self.log.skip(f"{input_path.name}: {msg}")
            return False, 0.0, msg

        total_sec = len(mono_samples) / framerate
        if total_sec < self.min_duration:
            msg = f"Duration {total_sec:.2f}s is below minimum threshold ({self.min_duration:.1f}s)"
            return False, total_sec, msg
        if total_sec > self.max_duration:
            msg = f"Duration {total_sec:.2f}s exceeds maximum threshold ({self.max_duration:.1f}s)"
            return False, total_sec, msg

        # Energy-based silence trimming
        threshold = 32767 * self.db_to_linear(self.silence_thresh_db)
        window_size = max(1, int(framerate * (self.window_size_ms / 1000.0)))

        start_idx = 0
        while start_idx < len(mono_samples):
            chunk = mono_samples[start_idx : start_idx + window_size]
            if self.calculate_rms(chunk) > threshold:
                break
            start_idx += window_size

        end_idx = len(mono_samples)
        while end_idx > start_idx:
            chunk = mono_samples[max(start_idx, end_idx - window_size) : end_idx]
            if self.calculate_rms(chunk) > threshold:
                break
            end_idx -= window_size

        trimmed = mono_samples[start_idx:end_idx]
        if not trimmed:
            msg = "Silence trimming stripped all audio samples"
            return False, 0.0, msg

        trimmed_dur = len(trimmed) / framerate
        if trimmed_dur < self.min_duration:
            msg = f"Trimmed duration {trimmed_dur:.2f}s is below minimum duration"
            return False, trimmed_dur, msg

        # Peak Normalization (e.g. 90% full scale ~ -1 dBFS)
        max_val = max(abs(s) for s in trimmed) or 1
        target_peak = int(32767 * self.norm_peak_ratio)
        norm_factor = target_peak / max_val
        normalized = [max(-32768, min(32767, int(s * norm_factor))) for s in trimmed]

        # Write output WAV file safely
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as out_wf:
                out_wf.setnchannels(1)
                out_wf.setsampwidth(2)
                out_wf.setframerate(framerate)
                packed = struct.pack(f"<{len(normalized)}h", *normalized)
                out_wf.writeframes(packed)
        except Exception as exc:
            msg = f"Failed writing output file {output_path}: {exc}"
            self.log.error(msg)
            return False, 0.0, msg

        final_duration = len(normalized) / framerate
        return True, final_duration, "OK"

    def batch_process(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Process all WAV files in input_dir and export XTTS/LJSpeech metadata.csv.
        Returns summary dictionary.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        self.log.info("Starting Audio Dataset Preprocessing...")
        self.log.info(f"Input Directory:  {input_dir}")
        self.log.info(f"Output Directory: {output_dir}")

        if not input_dir.exists() or not input_dir.is_dir():
            self.log.error(f"Input directory does not exist: {input_dir}")
            return {
                "total": 0,
                "accepted": 0,
                "rejected": 0,
                "total_sec": 0.0,
                "metadata_csv": None,
            }

        wav_files = sorted(list(input_dir.glob("**/*.wav")))
        self.log.info(f"Discovered {len(wav_files)} WAV files.")

        metadata: List[Dict[str, Any]] = []
        accepted = 0
        rejected = 0
        total_sec = 0.0

        for idx, wav in enumerate(wav_files, 1):
            out_file = output_dir / f"clean_{wav.name}"
            ok, duration, reason = self.process_wav(wav, out_file)
            if ok:
                accepted += 1
                total_sec += duration
                rel_path = out_file.name
                metadata.append({"audio_file": rel_path, "duration": round(duration, 2)})
                self.log.success(f"[{idx}/{len(wav_files)}] {wav.name} ({duration:.2f}s)")
            else:
                rejected += 1
                self.log.warning(f"[{idx}/{len(wav_files)}] REJECTED: {wav.name} -> {reason}")

        output_dir.mkdir(parents=True, exist_ok=True)
        meta_csv = output_dir / "metadata.csv"
        try:
            with open(meta_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["audio_file", "duration"])
                writer.writeheader()
                writer.writerows(metadata)
        except Exception as exc:
            self.log.error(f"Failed writing metadata.csv: {exc}")

        print("\n" + "=" * 55)
        self.log.info(f"Total Processed:         {len(wav_files)}")
        self.log.success(f"Accepted Clean Samples:  {accepted}")
        self.log.warning(f"Rejected Samples:        {rejected}")
        self.log.info(f"Total Clean Audio Time:  {total_sec / 60:.2f} minutes ({total_sec:.2f} seconds)")
        self.log.info(f"Metadata Manifest:       {meta_csv}")
        print("=" * 55 + "\n")

        return {
            "total": len(wav_files),
            "accepted": accepted,
            "rejected": rejected,
            "total_sec": total_sec,
            "metadata_csv": str(meta_csv),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean, trim silence, normalize and prepare audio datasets for TTS / Voice Cloning models."
    )
    parser.add_argument("--input", "-i", type=str, default=None, help="Input directory containing raw audio files")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory for clean dataset")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Silence threshold in dB (e.g. -40.0)")
    parser.add_argument("--min-dur", type=float, default=None, help="Minimum audio duration in seconds")
    parser.add_argument("--max-dur", type=float, default=None, help="Maximum audio duration in seconds")
    parser.add_argument("--norm-peak", type=float, default=None, help="Peak normalization ratio (0.0 to 1.0)")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to JSON configuration file")
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env file")
    parser.add_argument("--no-color", action="store_true", help="Disable colored console logging")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load defaults from config file and env file
    cfg_file = Path(args.config)
    env_file = Path(args.env_file)

    config_data = load_config_file(cfg_file)
    env_data = load_env_file(env_file)

    def resolve(cli_val, env_key, cfg_key, default_val, cast_fn=str):
        if cli_val is not None:
            return cast_fn(cli_val)
        if env_key in env_data:
            return cast_fn(env_data[env_key])
        if cfg_key in config_data:
            return cast_fn(config_data[cfg_key])
        return default_val

    input_dir = resolve(args.input, "INPUT_DIR", "input_dir", "raw_audio", str)
    output_dir = resolve(args.output, "OUTPUT_DIR", "output_dir", "dataset_clean", str)
    threshold = resolve(args.threshold, "SILENCE_THRESH_DB", "silence_thresh_db", -40.0, float)
    min_dur = resolve(args.min_dur, "MIN_DURATION", "min_duration", 1.0, float)
    max_dur = resolve(args.max_dur, "MAX_DURATION", "max_duration", 12.0, float)
    norm_peak = resolve(args.norm_peak, "NORM_PEAK_RATIO", "norm_peak_ratio", 0.9, float)

    custom_logger = Logger(color_enabled=not args.no_color)

    try:
        cleaner = AudioDatasetCleaner(
            silence_thresh_db=threshold,
            min_duration=min_dur,
            max_duration=max_dur,
            norm_peak_ratio=norm_peak,
            logger=custom_logger,
        )
        cleaner.batch_process(Path(input_dir), Path(output_dir))
    except Exception as exc:
        custom_logger.error(f"Fatal error during execution: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
