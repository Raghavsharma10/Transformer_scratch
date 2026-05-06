def spectrogram(t_signal, frame_width=FRAME_WIDTH, overlap=FRAME_STRIDE):
    """
    Calculate the magnitude spectrogram of a single-channel time-domain signal
    from the real frequency components of the STFT with a hanning window
    applied to each frame. The frame size and overlap between frames should
    be specified in number of samples.
    """
    frame_width = min(t_signal.shape[0], frame_width)
    w = np.hanning(frame_width)
    num_components = frame_width // 2 + 1
    num_frames = 1 + (len(t_signal) - frame_width) // overlap

    f_signal = np.empty([num_frames, num_components], dtype=np.complex_)
    for i, t in enumerate(range(0, len(t_signal) - frame_width, overlap)):
        # using rfft avoids computing negative frequency components
        f_signal[i] = rfft(w * t_signal[t:t + frame_width])

    # amplitude in decibels
    return 20 * np.log10(1 + np.absolute(f_signal))