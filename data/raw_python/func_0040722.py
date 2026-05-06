def orbfit_ephemeris(
        log,
        objectId,
        mjd,
        settings,
        obscode=500,
        verbose=False,
        astorbPath=False):
    """Given a known solar-system object ID (human-readable name or MPC number but *NOT* an MPC packed format) or list of names and one or more specific epochs, return the calculated ephemerides

    **Key Arguments:**
        - ``log`` -- logger
        - ``objectId`` -- human-readable name, MPC number or solar-system object name, or list of names
        - ``mjd`` -- a single MJD, or a list MJDs to generate an ephemeris for
        - ``settings`` -- the settings dictionary for rockfinder
        - ``obscode`` -- the observatory code for the ephemeris generation. Default **500** (geocentric)
        - ``verbose`` -- return extra information with each ephemeris
        - ``astorbPath`` -- override the default path to astorb.dat orbital elements file

    **Return:**
        - ``resultList`` -- a list of ordered dictionaries containing the returned ephemerides

    **Usage:**

        To generate a an ephemeris for a single epoch run,using ATLAS Haleakala as your observatory:

        .. code-block:: python

            from rockfinder import orbfit_ephemeris
            eph = orbfit_ephemeris(
                log=log,
                objectId=1,
                obscode="T05"
                mjd=57916.,
            )

        or to generate an ephemeris for multiple epochs:

        .. code-block:: python

            from rockfinder import orbfit_ephemeris
            eph = orbfit_ephemeris(
                log=log,
                objectId="ceres",
                mjd=[57916.1,57917.234,57956.34523]
                verbose=True
            )

        Note by passing `verbose=True` the essential ephemeris data is supplimented with some extra data

        It's also possible to pass in an array of object IDs:

        .. code-block:: python

            from rockfinder import orbfit_ephemeris
            eph = orbfit_ephemeris(
                log=log,
                objectId=[1,5,03547,"Shikoku"],
                mjd=[57916.1,57917.234,57956.34523]
            )

        And finally you override the default path to astorb.dat orbital elements file by passing in a custom path (useful for passing in a trimmed orbital elements database):

        .. code-block:: python

            from rockfinder import orbfit_ephemeris
            eph = orbfit_ephemeris(
                log=log,
                objectId=[1,5,03547,"Shikoku"],
                mjd=[57916.1,57917.234,57956.34523],
                astorbPath="/path/to/astorb.dat"
            )

    """
    log.debug('starting the ``orbfit_ephemeris`` function')

    global cmdList

    # MAKE SURE MJDs ARE IN A LIST
    if not isinstance(mjd, list):
        mjdList = [str(mjd)]
    else:
        mjdList = mjd

    if not isinstance(objectId, list):
        objectList = [objectId]
    else:
        objectList = objectId

    ephem = settings["path to ephem binary"]
    home = expanduser("~")

    results = []
    tmpCmdList = []

    for o in objectList:
        for m in mjdList:
            if not isinstance(o, int) and "'" in o:
                cmd = """%(ephem)s %(obscode)s %(m)s "%(o)s" """ % locals()
            else:
                cmd = """%(ephem)s %(obscode)s %(m)s '%(o)s'""" % locals()

            if astorbPath:
                cmd += " '%(astorbPath)s'" % locals()

            tmpCmdList.append((cmd, o))

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # BATCH INTO 10s
    cmdList = [c for c in chunks(tmpCmdList, 10)]

    # DEFINE AN INPUT ARRAY
    results = fmultiprocess(log=log, function=_generate_one_ephemeris,
                            inputArray=range(len(cmdList)))

    if verbose == True:
        order = ["object_name", "mjd", "ra_deg", "dec_deg", "apparent_mag", "observer_distance", "heliocentric_distance",
                 "phase_angle",  "obscode", "sun_obs_target_angle", "galactic_latitude", "ra_arcsec_per_hour", "dec_arcsec_per_hour"]
    else:
        order = ["object_name", "mjd", "ra_deg", "dec_deg", "apparent_mag", "observer_distance", "heliocentric_distance",
                 "phase_angle"]

    # ORDER THE RESULTS
    resultList = []
    for r2 in results:
        if not r2:
            continue
        for r in r2:
            if not r:
                continue

            orderDict = collections.OrderedDict({})
            for i in order:
                orderDict[i] = r[i]

            resultList.append(orderDict)

    log.debug('completed the ``orbfit_ephemeris`` function')
    return resultList