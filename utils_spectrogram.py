import numpy as np
import matplotlib.pyplot as plt
from maad.sound import load
from maad.sound import select_bandwidth
from maad.sound import spectrogram
from maad.util import amplitude2dB
from maad.util import power2dB


def get_spectrogram_image(
    # Audio file to operate on
    file_path,

    # Highpass filter parameters
    highpass_cutting_frequency=None,
    forder=3,

    # DB conversion parameters
    db_range=96,
    db_gain=0,

    # Spectrogram parameters
    frequency_crop=None,
    mode='complex',
    # mode='psd',
    # mode='amplitude',
    nperseg=512,
    noverlap=256,
    log_scale=True,
) -> np.ndarray:

    # Load the audio signal
    signal, sample_rate = load(str(file_path))

    # Apply a highpass filter
    if highpass_cutting_frequency is not None:
        signal = select_bandwidth(
            x=signal,
            fs=sample_rate,
            fcut=highpass_cutting_frequency,
            forder=forder,
            ftype='highpass',
        )

    # Calculate the spectrogram
    Sxx_power, tn, fn, extent = spectrogram(
        x=signal,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        fcrop=frequency_crop,
        log_scale=log_scale,
        mode=mode,
    )

    if mode == 'psd':
        Sxx_db = \
            power2dB(Sxx_power, db_range=db_range, db_gain=db_gain)
    if mode == 'amplitude':
        Sxx_db = \
            amplitude2dB(Sxx_power, db_range=db_range, db_gain=db_gain)
    if mode == 'complex':
        Sxx_db = \
            amplitude2dB(Sxx_power, db_range=db_range, db_gain=db_gain)

    # Convert the dB spectrogram to an RGB image
    cmap = plt.get_cmap('inferno')
    Sxx_rgb = cmap((Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db)))
    Sxx_rgb = (Sxx_rgb[:, :, :3] * 255).astype(np.uint8)

    return Sxx_rgb
