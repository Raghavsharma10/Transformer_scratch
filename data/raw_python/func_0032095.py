def onSiliconCheck(ra_deg, dec_deg, FovObj, padding_pix=DEFAULT_PADDING):
    """Check a single position."""
    dist = angSepVincenty(FovObj.ra0_deg, FovObj.dec0_deg, ra_deg, dec_deg)
    if dist >= 90.:
        return False
    # padding_pix=3 means that objects less than 3 pixels off the edge of
    # a channel are counted inside, to account for inaccuracies in K2fov.
    return FovObj.isOnSilicon(ra_deg, dec_deg, padding_pix=padding_pix)