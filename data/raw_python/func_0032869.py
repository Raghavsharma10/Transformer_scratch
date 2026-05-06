def findCampaigns(ra, dec):
    """Returns a list of the campaigns that cover a given position.

    Parameters
    ----------
    ra, dec : float, float
        Position in decimal degrees (J2000).

    Returns
    -------
    campaigns : list of int
        A list of the campaigns that cover the given position.
    """
    # Temporary disable the logger to avoid the preliminary field warnings
    logger.disabled = True
    campaigns_visible = []
    for c in fields.getFieldNumbers():
        fovobj = fields.getKeplerFov(c)
        if onSiliconCheck(ra, dec, fovobj):
            campaigns_visible.append(c)
    # Re-enable the logger
    logger.disabled = True
    return campaigns_visible