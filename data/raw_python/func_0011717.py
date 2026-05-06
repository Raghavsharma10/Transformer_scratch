def order_param(ts, axis=2):
    """Order parameter of phase synchronization"""
    return np.abs(np.exp(1.0j * ts).mean(axis=axis))