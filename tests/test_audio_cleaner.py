# -*- coding: utf-8 -*-
"""
Automated Unit Tests for Audio Dataset Cleaner
Author: Eng. MHD. Shadi AL-Hasan <mhd.shadi.alhasan@gmail.com>
"""

import os
import sys
import wave
import struct
import math
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_cleaner import (
    AudioDatasetCleaner,
    Logger,
    load_config_file,
    load_env_file,
    build_parser,
)


def create_test_wav(
    filepath: Path,
    duration_sec: float = 2.0,
    sample_rate: int = 22050,
    channels: int = 1,
    sample_width: int = 2,
    freq: float = 440.0,
    amplitude: float = 0.5,
    lead_silence_sec: float = 0.5,
    trail_silence_sec: float = 0.5,
):
    """Utility to generate synthetic test WAV files."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    total_samples = int(duration_sec * sample_rate)
    lead_samples = int(lead_silence_sec * sample_rate)
    trail_samples = int(trail_silence_sec * sample_rate)
    tone_samples = max(0, total_samples - lead_samples - trail_samples)

    raw_samples = []
    # Leading silence
    raw_samples.extend([0] * lead_samples)
    # Tone
    for i in range(tone_samples):
        val = int(32767 * amplitude * math.sin(2 * math.pi * freq * (i / sample_rate)))
        raw_samples.append(val)
    # Trailing silence
    raw_samples.extend([0] * trail_samples)

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        if channels == 1:
            packed = struct.pack(f"<{len(raw_samples)}h", *raw_samples)
        elif channels == 2:
            stereo = []
            for s in raw_samples:
                stereo.extend([s, s])
            packed = struct.pack(f"<{len(stereo)}h", *stereo)
        else:
            raise ValueError("Unsupported channel count in test helper")
        wf.writeframes(packed)


class TestAudioDatasetCleaner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_path = Path(self.temp_dir.name)
        self.cleaner = AudioDatasetCleaner(
            silence_thresh_db=-30.0,
            min_duration=0.5,
            max_duration=5.0,
            norm_peak_ratio=0.9,
            logger=Logger(color_enabled=False),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_calculate_rms(self):
        self.assertEqual(AudioDatasetCleaner.calculate_rms([]), 0.0)
        self.assertEqual(AudioDatasetCleaner.calculate_rms([0, 0, 0, 0]), 0.0)
        self.assertAlmostEqual(AudioDatasetCleaner.calculate_rms([10, 10, 10, 10]), 10.0)
        self.assertAlmostEqual(AudioDatasetCleaner.calculate_rms([10, -10, 10, -10]), 10.0)

    def test_db_to_linear(self):
        self.assertAlmostEqual(AudioDatasetCleaner.db_to_linear(0.0), 1.0)
        self.assertAlmostEqual(AudioDatasetCleaner.db_to_linear(-20.0), 0.1)
        self.assertAlmostEqual(AudioDatasetCleaner.db_to_linear(-40.0), 0.01)
        self.assertEqual(AudioDatasetCleaner.db_to_linear(-float("inf")), 0.0)

    def test_initialization_validation(self):
        with self.assertRaises(ValueError):
            AudioDatasetCleaner(min_duration=-1.0)
        with self.assertRaises(ValueError):
            AudioDatasetCleaner(min_duration=5.0, max_duration=2.0)
        with self.assertRaises(ValueError):
            AudioDatasetCleaner(target_sr=0)

    def test_process_wav_valid_mono(self):
        in_wav = self.work_path / "valid_mono.wav"
        out_wav = self.work_path / "out" / "clean_valid_mono.wav"
        create_test_wav(in_wav, duration_sec=2.0, lead_silence_sec=0.5, trail_silence_sec=0.5)

        ok, duration, msg = self.cleaner.process_wav(in_wav, out_wav)
        self.assertTrue(ok)
        self.assertTrue(out_wav.exists())
        self.assertGreater(duration, 0.5)
        self.assertLess(duration, 2.0)  # Silence should have been trimmed

        # Verify output WAV properties
        with wave.open(str(out_wav), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)

    def test_process_wav_stereo_to_mono(self):
        in_wav = self.work_path / "stereo.wav"
        out_wav = self.work_path / "out" / "clean_stereo.wav"
        create_test_wav(in_wav, duration_sec=2.0, channels=2)

        ok, duration, msg = self.cleaner.process_wav(in_wav, out_wav)
        self.assertTrue(ok)
        self.assertTrue(out_wav.exists())
        with wave.open(str(out_wav), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)

    def test_duration_filtering(self):
        # Clip too short
        short_wav = self.work_path / "too_short.wav"
        out_short = self.work_path / "out" / "clean_short.wav"
        create_test_wav(short_wav, duration_sec=0.3, lead_silence_sec=0.0, trail_silence_sec=0.0)
        ok, dur, msg = self.cleaner.process_wav(short_wav, out_short)
        self.assertFalse(ok)
        self.assertIn("minimum threshold", msg)

        # Clip too long
        long_wav = self.work_path / "too_long.wav"
        out_long = self.work_path / "out" / "clean_long.wav"
        create_test_wav(long_wav, duration_sec=6.0, lead_silence_sec=0.0, trail_silence_sec=0.0)
        ok, dur, msg = self.cleaner.process_wav(long_wav, out_long)
        self.assertFalse(ok)
        self.assertIn("exceeds maximum threshold", msg)

    def test_nonexistent_and_corrupt_files(self):
        nonexistent = self.work_path / "does_not_exist.wav"
        ok, _, msg = self.cleaner.process_wav(nonexistent, self.work_path / "out.wav")
        self.assertFalse(ok)

        corrupt = self.work_path / "corrupt.wav"
        corrupt.write_bytes(b"NOT A REAL WAV FILE DATA HEADER")
        ok, _, msg = self.cleaner.process_wav(corrupt, self.work_path / "out_corrupt.wav")
        self.assertFalse(ok)

    def test_batch_process_and_metadata(self):
        raw_dir = self.work_path / "raw"
        out_dir = self.work_path / "clean"
        raw_dir.mkdir(parents=True, exist_ok=True)

        create_test_wav(raw_dir / "clip1.wav", duration_sec=2.0)
        create_test_wav(raw_dir / "clip2.wav", duration_sec=3.0)
        create_test_wav(raw_dir / "short.wav", duration_sec=0.2)  # Will be rejected

        summary = self.cleaner.batch_process(raw_dir, out_dir)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["rejected"], 1)

        meta_csv = out_dir / "metadata.csv"
        self.assertTrue(meta_csv.exists())
        content = meta_csv.read_text(encoding="utf-8")
        self.assertIn("audio_file,duration", content)
        self.assertIn("clean_clip1.wav", content)
        self.assertIn("clean_clip2.wav", content)
        self.assertNotIn("clean_short.wav", content)

    def test_batch_process_empty_or_missing_dir(self):
        missing_dir = self.work_path / "no_such_dir"
        out_dir = self.work_path / "clean_empty"
        summary = self.cleaner.batch_process(missing_dir, out_dir)
        self.assertEqual(summary["total"], 0)

    def test_config_and_env_loaders(self):
        cfg_path = self.work_path / "test_config.json"
        cfg_path.write_text('{"target_sr": 16000, "silence_thresh_db": -35.0}', encoding="utf-8")
        loaded_cfg = load_config_file(cfg_path)
        self.assertEqual(loaded_cfg["target_sr"], 16000)
        self.assertEqual(loaded_cfg["silence_thresh_db"], -35.0)

        env_path = self.work_path / ".env.test"
        env_path.write_text("MIN_DURATION=1.5\n# Comment\nMAX_DURATION=15.0\n", encoding="utf-8")
        loaded_env = load_env_file(env_path)
        self.assertEqual(loaded_env["MIN_DURATION"], "1.5")
        self.assertEqual(loaded_env["MAX_DURATION"], "15.0")

    def test_parser(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "my_in", "--output", "my_out", "--threshold", "-35.5"])
        self.assertEqual(args.input, "my_in")
        self.assertEqual(args.output, "my_out")
        self.assertEqual(args.threshold, -35.5)


if __name__ == "__main__":
    unittest.main()
