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
import wave
import struct
import math
import argparse
from pathlib import Path

class AudioDatasetCleaner:
    def __init__(self, target_sr=22050, silence_thresh_db=-40.0, min_duration=1.0, max_duration=12.0):
        self.target_sr = target_sr
        self.silence_thresh_db = silence_thresh_db
        self.min_duration = min_duration
        self.max_duration = max_duration

    def calculate_rms(self, samples):
        if not samples:
            return 0.0
        sum_squares = sum(s ** 2 for s in samples)
        return math.sqrt(sum_squares / len(samples))

    def db_to_linear(self, db):
        return 10.0 ** (db / 20.0)

    def process_wav(self, input_path: Path, output_path: Path):
        """Processes a single WAV file: trims silence, normalizes amplitude, and verifies duration."""
        try:
            with wave.open(str(input_path), "rb") as wf:
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                num_frames = wf.getnframes()
                raw_data = wf.readframes(num_frames)

            # Support 16-bit PCM
            if sample_width != 2:
                print(f"[SKIP] {input_path.name}: Only 16-bit PCM supported directly.")
                return False, 0.0

            fmt = f"<{num_frames * num_channels}h"
            samples = list(struct.unpack(fmt, raw_data))

            # Convert stereo to mono if needed
            if num_channels == 2:
                mono_samples = [(samples[i] + samples[i+1]) // 2 for i in range(0, len(samples), 2)]
            else:
                mono_samples = samples

            total_sec = len(mono_samples) / framerate
            if total_sec < self.min_duration or total_sec > self.max_duration:
                return False, total_sec

            # Basic energy-based silence trimming
            threshold = 32767 * self.db_to_linear(self.silence_thresh_db)
            window_size = int(framerate * 0.02) # 20ms window
            
            start_idx = 0
            while start_idx < len(mono_samples):
                chunk = mono_samples[start_idx : start_idx + window_size]
                if self.calculate_rms(chunk) > threshold:
                    break
                start_idx += window_size

            end_idx = len(mono_samples)
            while end_idx > start_idx:
                chunk = mono_samples[end_idx - window_size : end_idx]
                if self.calculate_rms(chunk) > threshold:
                    break
                end_idx -= window_size

            trimmed = mono_samples[start_idx:end_idx]
            if not trimmed:
                return False, 0.0

            # Peak Normalization to -1 dBFS (~90% max amplitude)
            max_val = max(abs(s) for s in trimmed) or 1
            target_peak = int(32767 * 0.9)
            norm_factor = target_peak / max_val
            normalized = [int(s * norm_factor) for s in trimmed]

            # Save normalized mono WAV
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as out_wf:
                out_wf.setnchannels(1)
                out_wf.setsampwidth(2)
                out_wf.setframerate(framerate)
                packed = struct.pack(f"<{len(normalized)}h", *normalized)
                out_wf.writeframes(packed)

            final_duration = len(normalized) / framerate
            return True, final_duration

        except Exception as e:
            print(f"[ERROR] Failed processing {input_path.name}: {e}")
            return False, 0.0

    def batch_process(self, input_dir: Path, output_dir: Path):
        print(f"[*] Starting Audio Dataset Preprocessing...")
        print(f"[*] Input Directory: {input_dir}")
        print(f"[*] Output Directory: {output_dir}")

        wav_files = list(input_dir.glob("**/*.wav"))
        print(f"[*] Found {len(wav_files)} WAV files.")

        metadata = []
        accepted = 0
        total_sec = 0.0

        for idx, wav in enumerate(wav_files, 1):
            out_file = output_dir / f"clean_{wav.name}"
            ok, duration = self.process_wav(wav, out_file)
            if ok:
                accepted += 1
                total_sec += duration
                # XTTS / LJSpeech style metadata format
                rel_path = f"clean_{wav.name}"
                metadata.append({"audio_file": rel_path, "duration": round(duration, 2)})
                print(f"[{idx}/{len(wav_files)}] OK: {wav.name} ({duration:.2f}s)")
            else:
                print(f"[{idx}/{len(wav_files)}] REJECTED: {wav.name}")

        # Write metadata.csv
        meta_csv = output_dir / "metadata.csv"
        with open(meta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_file", "duration"])
            writer.writeheader()
            writer.writerows(metadata)

        print("\n" + "=" * 50)
        print(f"[SUMMARY] Total Processed: {len(wav_files)}")
        print(f"[SUMMARY] Accepted Clean Samples: {accepted}")
        print(f"[SUMMARY] Total Clean Audio Duration: {total_sec / 60:.2f} minutes")
        print(f"[SUMMARY] Metadata saved to: {meta_csv}")
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Clean and prepare audio datasets for TTS / Voice Cloning models.")
    parser.add_argument("--input", "-i", type=str, default="raw_audio", help="Input folder containing raw audio files")
    parser.add_argument("--output", "-o", type=str, default="dataset_clean", help="Output directory for clean dataset")
    parser.add_argument("--threshold", "-t", type=float, default=-40.0, help="Silence threshold in dB")
    parser.add_argument("--min-dur", type=float, default=1.0, help="Minimum duration in seconds")
    parser.add_argument("--max-dur", type=float, default=12.0, help="Maximum duration in seconds")
    args = parser.parse_args()

    cleaner = AudioDatasetCleaner(
        silence_thresh_db=args.threshold,
        min_duration=args.min_dur,
        max_duration=args.max_dur
    )
    cleaner.batch_process(Path(args.input), Path(args.output))

if __name__ == "__main__":
    main()
