def circmean(dts, axis=2):
    """Circular mean phase"""
    return np.exp(1.0j * dts).mean(axis=axis).angle()