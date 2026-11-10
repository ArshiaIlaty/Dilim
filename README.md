# azb_subtitles

Test the community **South Azerbaijani (Iranian Azeri / Turki)** speech models on your own recording and get a subtitle file.

The models are [Kartal-Ol/ASR-AZB](https://huggingface.co/Kartal-Ol/ASR-AZB), from the INTERSPEECH 2026 paper *Preserving the Iranian Turkic Language*. They run free on your own computer.

## Setup (Mac, once)

```bash
brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Use

```bash
python azb_subtitles.py my_video.mov                 # quick: whisper-base-full (~300 MB)
python azb_subtitles.py my_video.mov --compare       # tries 3 models, one .srt each
python azb_subtitles.py my_video.mov --model whisper-small-north-azerbaijani
```

It writes `my_video.<model>.srt`, which you can import into CapCut, and `my_video.<model>.txt` for reading. The first run downloads the model.

## What to expect

On 24 sample sentences from an open South Azerbaijani dataset:

| Model | Word errors | Letter errors | Speed |
| --- | --- | --- | --- |
| whisper-base-full, as released | 52% | 46% | fast |
| whisper-base-full + this script's guard | 28% | 16% | fast |
| whisper-Small + guard | 18% | 8% | ~4x slower |
| whisper-small-north-azerbaijani + guard | 19% | 8% | ~4x slower |

These numbers are **optimistic**. Most of those sentences, and probably the same recordings, were in the models' training data. Read-aloud book sentences are also easier than natural speech. **Your own voice is the real test.**

Many of the "word errors" are only spacing, like `گؤزله‌دی` vs `گؤزله دی`, because Azeri spelling in Arabic script isn't standardized. The letter error rate shows readability better.

## The guard (why this script, not the model card's example)

As released, the models often **don't stop**: after the correct sentence they repeat it or loop a word. The script:
1. Splits your audio at pauses into short pieces.
2. Caps each piece's output length by how long the speech is.
3. Blocks word loops.
4. Cuts the text where the model restarts the sentence.

## Limits

- **Phrase-level timing.** Subtitles are timed per phrase, which is fine for captions. For word-exact timing (needed by `autooverlay`), the model would need fine-tuning that keeps timestamp tokens.
- **No license on the model card.** Personal testing is fine; ask the authors before redistributing or building a product on these models.
