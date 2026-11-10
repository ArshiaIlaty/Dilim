#!/usr/bin/env python3
"""
azb_subtitles: try the community South Azerbaijani (Iranian Azeri / Turki) Whisper
models on your own recording, and get a subtitle file (.srt).

Models: Kartal-Ol/ASR-AZB on Hugging Face (from the INTERSPEECH 2026 paper
"Preserving the Iranian Turkic Language"). Free and open; runs on your computer.

Usage:
  python azb_subtitles.py my_video.mp4                     # default model (whisper-base-full)
  python azb_subtitles.py my_video.mp4 --model whisper-small-north-azerbaijani
  python azb_subtitles.py my_video.mp4 --compare           # try several models, one .srt each

Output: my_video.<model>.srt and my_video.<model>.txt next to your video.

Why the extra "guard" code: these checkpoints transcribe Azeri words well but often
don't stop at the end of a sentence. They repeat it or add unrelated words. The guard
cuts each piece at the point where it starts repeating, caps the length by how long
the speech is, and blocks word loops.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "Kartal-Ol/ASR-AZB"
MODELS = ["whisper-base-full", "whisper-small-north-azerbaijani", "whisper-Small",
          "whisper-Small-Farsi", "whisper-small-turkish", "whisper-base", "whisper-tiny"]
COMPARE = ["whisper-base-full", "whisper-small-north-azerbaijani", "whisper-Small"]
SR = 16000


# ---------------------------------------------------------------- audio
def load_audio(path: Path):
    import numpy as np
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. Mac: brew install ffmpeg")
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", str(SR), "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def speech_chunks(audio, max_len: float = 12.0, min_len: float = 0.6):
    """Split audio at pauses into pieces of at most `max_len` seconds (Silero VAD via faster-whisper)."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=350, speech_pad_ms=200,
                                                     max_speech_duration_s=max_len))
        spans = [(t["start"] / SR, t["end"] / SR) for t in ts]
    except Exception:
        dur = len(audio) / SR  # fallback: fixed windows
        spans = [(s, min(s + max_len, dur)) for s in [i * max_len for i in range(int(dur // max_len) + 1)]]
    # merge very short neighbours so each piece has some context
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and (e - merged[-1][0] <= max_len) and (s - merged[-1][1] < 0.4 or e - s < min_len):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= 0.3]


# ---------------------------------------------------------------- the guard
def _restart_cut(text: str) -> str:
    """Cut where the model starts the sentence over, even if the restart is glued to the
    previous word ("...ائوهابولقاسیم") or slightly reworded. Compares letters, ignoring spaces."""
    from difflib import SequenceMatcher
    pos = [i for i, ch in enumerate(text) if not ch.isspace()]   # letter index -> text index
    s = "".join(text[i] for i in pos)
    if len(s) < 24:
        return text
    probe = s[:4]
    start = int(len(s) * 0.35)                                    # a restart comes after real content
    p = s.find(probe, start)
    while p != -1:
        L = min(len(s) - p, 40)
        if L >= 8 and SequenceMatcher(None, s[p:p + L], s[:L]).ratio() >= 0.6:
            return text[:pos[p]].rstrip(" ،,")
        p = s.find(probe, p + 1)
    return text


def cut_repetition(text: str) -> str:
    """Remove the two failure patterns: restarting the sentence, and looping a short phrase."""
    words = _restart_cut(text).split()
    # a short phrase looping at the end: "... اوّل اوّل اوّل" / "هه، هه، هه"
    for n in (1, 2, 3):
        while len(words) >= 3 * n and words[-n:] == words[-2 * n:-n] == words[-3 * n:-2 * n]:
            words = words[:-n]
        while len(words) >= 2 * n and words[-n:] == words[-2 * n:-n] and n > 1:
            words = words[:-n]
    return " ".join(words)


class Transcriber:
    def __init__(self, subfolder: str):
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        print(f"[model] loading {REPO}/{subfolder} (first time downloads it)")
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(REPO, subfolder=subfolder)
        self.model = WhisperForConditionalGeneration.from_pretrained(REPO, subfolder=subfolder).eval()
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

    def __call__(self, piece) -> str:
        dur = len(piece) / SR
        feats = self.proc(piece, sampling_rate=SR, return_tensors="pt").input_features.to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                feats,
                max_new_tokens=min(220, int(12 * dur) + 12),  # cap by speech length
                no_repeat_ngram_size=4,                      # block word loops
                num_beams=3,
            )
        text = self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        return cut_repetition(text)


# ---------------------------------------------------------------- srt
def ts(t: float) -> str:
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3_600_000); m, ms = divmod(ms, 60_000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_srt(items: list[tuple[float, float, str]], max_words: int = 8) -> str:
    blocks, k = [], 0
    for s, e, text in items:
        words = text.split()
        if not words:
            continue
        parts = [words[i:i + max_words] for i in range(0, len(words), max_words)]
        step = (e - s) / len(parts)  # spread long pieces evenly over their time span
        for j, p in enumerate(parts):
            k += 1
            blocks.append(f"{k}\n{ts(s + j * step)} --> {ts(s + (j + 1) * step)}\n{' '.join(p)}\n")
    return "\n".join(blocks)


def run(video: Path, subfolder: str, audio, chunks) -> Path:
    tr = Transcriber(subfolder)
    items = []
    for s, e in chunks:
        text = tr(audio[int(s * SR):int(e * SR)])
        items.append((s, e, text))
        print(f"  {s:6.1f}-{e:6.1f}s  {text}")
    srt = video.with_suffix(f".{subfolder}.srt")
    srt.write_text(to_srt(items), encoding="utf-8")
    video.with_suffix(f".{subfolder}.txt").write_text("\n".join(t for _, _, t in items), encoding="utf-8")
    print(f"[done] {srt}")
    return srt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--model", default="whisper-base-full", choices=MODELS)
    ap.add_argument("--compare", action="store_true", help=f"run {', '.join(COMPARE)}")
    a = ap.parse_args()
    if not a.video.exists():
        sys.exit(f"Not found: {a.video}")
    audio = load_audio(a.video)
    chunks = speech_chunks(audio)
    print(f"[audio] {len(audio)/SR:.1f}s, {len(chunks)} speech pieces")
    for m in (COMPARE if a.compare else [a.model]):
        run(a.video, m, audio, chunks)


if __name__ == "__main__":
    main()
