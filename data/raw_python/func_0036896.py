def log_magnitude_spectrum(frames):
    """Compute the log of the magnitude spectrum of frames"""
    return N.log(N.abs(N.fft.rfft(frames)).clip(1e-5, N.inf))