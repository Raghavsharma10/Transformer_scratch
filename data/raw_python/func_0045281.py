def _get_hash(f1, f2, dt):
    """
    Calculate a 64-bit integral hash from <f1:f2:dt>, where f1 and f2 are
    FFT frequency bins (based on frame width), and dt is propotional to the
    time difference between f1 and f2 as the the difference in frame number
    between the points.
    """
    return ((long(f1) & 0xffff) << 48 |
            (long(f2) & 0xffff) << 32 |
            (long(dt) & 0x3fff))