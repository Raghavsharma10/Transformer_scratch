def delta_hv(scatterer):
    """
    Delta_hv for the current setup.

    Args:
        scatterer: a Scatterer instance.

    Returns:
        Delta_hv [rad].
    """
    Z = scatterer.get_Z()
    return np.arctan2(Z[2,3] - Z[3,2], -Z[2,2] - Z[3,3])