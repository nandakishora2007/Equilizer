import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np


def extract_audio(video_path):
    """
    Extract mono 16 kHz WAV audio using FFmpeg.

    Returns:
        Path to temporary WAV file, or None if no audio exists.
    """

    temp_dir = Path(
        tempfile.mkdtemp(prefix="deepfake_audio_")
    )

    audio_path = temp_dir / "audio.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(audio_path)
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

    except FileNotFoundError as exc:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and add "
            "it to your system PATH."
        ) from exc

    if (
        result.returncode != 0
        or not audio_path.exists()
        or audio_path.stat().st_size == 0
    ):
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )
        return None

    return audio_path


def load_audio(video_path):
    """
    Extract and load audio.

    Returns:
        (audio, sample_rate, temporary_audio_path)
    """

    audio_path = extract_audio(video_path)

    if audio_path is None:
        return None, None, None

    try:
        audio, sample_rate = librosa.load(
            str(audio_path),
            sr=16000,
            mono=True
        )

        if audio.size == 0:
            cleanup_audio(audio_path)
            return None, None, None

        return audio, sample_rate, audio_path

    except Exception:
        cleanup_audio(audio_path)
        raise


def cleanup_audio(audio_path):
    """
    Delete the temporary WAV file and its temporary directory.
    """

    if audio_path is None:
        return

    audio_path = Path(audio_path)

    # The temporary directory is created specifically for this audio.
    shutil.rmtree(
        audio_path.parent,
        ignore_errors=True
    )


def audio_features(audio, sample_rate):
    """
    Extract lightweight audio features.

    These are supporting heuristics, NOT a trained
    scientific audio deepfake detector.
    """

    if (
        audio is None
        or len(audio) == 0
        or sample_rate is None
    ):
        return None

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    spectral_flatness = librosa.feature.spectral_flatness(
        y=audio
    )

    return {
        "mfcc_mean": float(np.mean(mfcc)),
        "mfcc_std": float(np.std(mfcc)),
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(
            np.mean(spectral_centroid)
        ),
        "spectral_flatness_mean": float(
            np.mean(spectral_flatness)
        )
    }