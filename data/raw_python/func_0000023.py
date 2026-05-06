def fix_crinfo(crinfo, to="axis"):
    """
    Function recognize order of crinfo and convert it to proper format.
    """

    crinfo = np.asarray(crinfo)
    if crinfo.shape[0] == 2:
        crinfo = crinfo.T

    return crinfo