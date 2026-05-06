def cwtmorlet(points, width):
    """complex morlet wavelet function compatible with scipy.signal.cwt
    Parameters: points: int
                    Number of points in `vector`.
                width: scalar
                    Width parameter of wavelet.
                    Equals (sample rate / fundamental frequency of wavelet)
    Returns: `vector`: complex-valued ndarray of shape (points,)
    """
    omega = 5.0
    s = points / (2.0 * omega * width)
    return wavelets.morlet(points, omega, s, complete=True)