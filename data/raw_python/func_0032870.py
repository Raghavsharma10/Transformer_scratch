def findCampaignsByName(target):
    """Returns a list of the campaigns that cover a given target.

    Parameters
    ----------
    target : str
        Name of the celestial object.

    Returns
    -------
    campaigns : list of int
        A list of the campaigns that cover the given target name.

    ra, dec : float, float
        Resolved coordinates in decimal degrees (J2000).

    Exceptions
    ----------
    Raises an ImportError if AstroPy is not installed.
    Raises a ValueError if `name` cannot be resolved to coordinates.
    """
    # Is AstroPy (optional dependency) installed?
    try:
        from astropy.coordinates import SkyCoord
        from astropy.coordinates.name_resolve import NameResolveError
        from astropy.utils.data import conf
        conf.remote_timeout = 90
    except ImportError:
        print('Error: AstroPy needs to be installed for this feature.')
        sys.exit(1)
    # Translate the target name into celestial coordinates
    try:
        crd = SkyCoord.from_name(target)
    except NameResolveError:
        raise ValueError('Could not find coordinates '
                         'for target "{0}".'.format(target))
    # Find the campaigns with visibility
    return findCampaigns(crd.ra.deg, crd.dec.deg), crd.ra.deg, crd.dec.deg