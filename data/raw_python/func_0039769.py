def calc_uncertainty(quantity, sys_unc, mean=True):
    """Calculate the combined standard uncertainty of a quantity."""
    n = len(quantity)
    std = np.nanstd(quantity)
    if mean:
        std /= np.sqrt(n)
    return np.sqrt(std**2 + sys_unc**2)