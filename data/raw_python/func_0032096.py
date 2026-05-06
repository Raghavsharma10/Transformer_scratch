def onSiliconCheckList(ra_deg, dec_deg, FovObj, padding_pix=DEFAULT_PADDING):
    """Check a list of positions."""
    dist = angSepVincenty(FovObj.ra0_deg, FovObj.dec0_deg, ra_deg, dec_deg)
    mask = (dist < 90.)
    out = np.zeros(len(dist), dtype=bool)
    out[mask] = FovObj.isOnSiliconList(ra_deg[mask], dec_deg[mask], padding_pix=padding_pix)
    return out