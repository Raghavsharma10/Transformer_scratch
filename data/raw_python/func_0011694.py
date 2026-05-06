def order_param(dts, axis=2):
    """Order parameter of phase synchronization"""
    return np.abs(np.exp(1.0j * dts).mean(axis=axis))