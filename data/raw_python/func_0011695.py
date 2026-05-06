def circstd(dts, axis=2):
    """Circular standard deviation"""
    R = np.abs(np.exp(1.0j * dts).mean(axis=axis))
    return np.sqrt(-2.0 * np.log(R))