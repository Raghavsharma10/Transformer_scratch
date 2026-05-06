def scope2lat(telescope):
    """
    Convert a telescope name into a latitude
    returns None when the telescope is unknown.

    Parameters
    ----------
    telescope : str
        Acronym (name) of telescope, eg MWA.

    Returns
    -------
    lat : float
        The latitude of the telescope.

    Notes
    -----
    These values were taken from wikipedia so have varying precision/accuracy
    """
    scopes = {'MWA': -26.703319,
              "ATCA": -30.3128,
              "VLA": 34.0790,
              "LOFAR": 52.9088,
              "KAT7": -30.721,
              "MEERKAT": -30.721,
              "PAPER": -30.7224,
              "GMRT": 19.096516666667,
              "OOTY": 11.383404,
              "ASKAP": -26.7,
              "MOST": -35.3707,
              "PARKES": -32.999944,
              "WSRT": 52.914722,
              "AMILA": 52.16977,
              "AMISA": 52.164303,
              "ATA": 40.817,
              "CHIME": 49.321,
              "CARMA": 37.28044,
              "DRAO": 49.321,
              "GBT": 38.433056,
              "LWA": 34.07,
              "ALMA": -23.019283,
              "FAST": 25.6525
              }
    if telescope.upper() in scopes:
        return scopes[telescope.upper()]
    else:
        log = logging.getLogger("Aegean")
        log.warn("Telescope {0} is unknown".format(telescope))
        log.warn("integrated fluxes may be incorrect")
        return None