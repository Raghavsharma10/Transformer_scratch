def getKeplerFov(fieldnum):
    """Returns a `fov.KeplerFov` object for a given campaign.

    Parameters
    ----------
    fieldnum : int
        K2 Campaign number.

    Returns
    -------
    fovobj : `fov.KeplerFov` object
        Details the footprint of the requested K2 campaign.
    """
    info = getFieldInfo(fieldnum)
    ra, dec, scRoll = info["ra"], info["dec"], info["roll"]
    # convert from SC roll to FOV coordinates
    # do not use the fovRoll coords anywhere else
    # they are internal to this script only
    fovRoll = fov.getFovAngleFromSpacecraftRoll(scRoll)
    # KeplerFov takes a listen of broken CCD channels as optional argument;
    # these channels will be ignored during plotting and on-silicon determination.
    # Modules 3 and 7 broke prior to the start of K2:
    brokenChannels = [5, 6, 7, 8,  17, 18, 19, 20]
    # Module 4 failed during Campaign 10
    if fieldnum > 10:
        brokenChannels.extend([9, 10, 11, 12])
    # Hack: the Kepler field is defined as "Campaign 1000"
    # and (initially) had no broken channels
    if fieldnum == 1000:
        brokenChannels = []

    return fov.KeplerFov(ra, dec, fovRoll, brokenChannels=brokenChannels)