def circmean(ts, axis=2):
    """Circular mean phase"""
    return np.exp(1.0j * ts).mean(axis=axis).angle()