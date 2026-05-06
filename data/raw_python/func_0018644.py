def get_generic_rcp_name(inname):
    """Convert an RCP name into the generic Pymagicc RCP name

    The conversion is case insensitive.

    Parameters
    ----------
    inname : str
        The name for which to get the generic Pymagicc RCP name

    Returns
    -------
    str
        The generic Pymagicc RCP name

    Examples
    --------
    >>> get_generic_rcp_name("RCP3PD")
    "rcp26"
    """
    # TODO: move into OpenSCM
    mapping = {
        "rcp26": "rcp26",
        "rcp3pd": "rcp26",
        "rcp45": "rcp45",
        "rcp6": "rcp60",
        "rcp60": "rcp60",
        "rcp85": "rcp85",
    }
    try:
        return mapping[inname.lower()]
    except KeyError:
        error_msg = "No generic name for input: {}".format(inname)
        raise ValueError(error_msg)