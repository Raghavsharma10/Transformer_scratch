def _mix_simple(a, b, x):
    """Mix b (with proportion x) with a."""
    x = np.clip(x, 0.0, 1.0)
    return (1.0 - x)*a + x*b