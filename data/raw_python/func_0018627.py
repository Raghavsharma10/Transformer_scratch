def convert_magicc_to_openscm_regions(regions, inverse=False):
    """
    Convert MAGICC regions to OpenSCM regions

    Parameters
    ----------
    regions : list_like, str
        Regions to convert

    inverse : bool
        If True, convert the other way i.e. convert OpenSCM regions to MAGICC7
        regions

    Returns
    -------
    ``type(regions)``
        Set of converted regions
    """
    if isinstance(regions, (list, pd.Index)):
        return [_apply_convert_magicc_to_openscm_regions(r, inverse) for r in regions]
    else:
        return _apply_convert_magicc_to_openscm_regions(regions, inverse)