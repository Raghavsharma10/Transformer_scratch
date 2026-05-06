def check_for_soliton(img_id):
    """Workhorse function.

    Creates the polynom.
    Calculates radius constraints from attributes in `ringcube` object.

    Parameters
    ----------
    ringcube : pyciss.ringcube.RingCube
        A containter class for a ring-projected ISS image file.

    Returns
    -------
    dict
        Dictionary with all solitons found. Reason why it is a dict is
        that it could be more than one in one image.
    """
    pm = io.PathManager(img_id)
    try:
        ringcube = RingCube(pm.cubepath)
    except FileNotFoundError:
        ringcube = RingCube(pm.undestriped)
    polys = create_polynoms()
    minrad = ringcube.minrad.to(u.km)
    maxrad = ringcube.maxrad.to(u.km)
    delta_years = get_year_since_resonance(ringcube)
    soliton_radii = {}
    for k, p in polys.items():
        current_r = p(delta_years) * u.km
        if minrad < current_r < maxrad:
            soliton_radii[k] = current_r
    return soliton_radii if soliton_radii else None