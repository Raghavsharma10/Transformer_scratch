def RMS_energy(frames):
    """Computes the RMS energy of frames"""
    f = frames.flatten()
    return N.sqrt(N.mean(f * f))