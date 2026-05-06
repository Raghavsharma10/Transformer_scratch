def GetNeighbors(EPIC, season=None, model=None, neighbors=10,
                 mag_range=(11., 13.),
                 cdpp_range=None, aperture_name='k2sff_15',
                 cadence='lc', **kwargs):
    '''
    Return `neighbors` random bright stars on the same module as `EPIC`.

    :param int EPIC: The EPIC ID number
    :param str model: The :py:obj:`everest` model name. Only used when \
           imposing CDPP bounds. Default :py:obj:`None`
    :param int neighbors: Number of neighbors to return. Default 10
    :param str aperture_name: The name of the aperture to use. Select \
           `custom` to call \
           :py:func:`GetCustomAperture`. Default `k2sff_15`
    :param str cadence: The light curve cadence. Default `lc`
    :param tuple mag_range: (`low`, `high`) values for the Kepler magnitude. \
           Default (11, 13)
    :param tuple cdpp_range: (`low`, `high`) values for the de-trended CDPP. \
           Default :py:obj:`None`

    '''

    # Zero neighbors?
    if neighbors == 0:
        return []

    # Get the IDs
    # Campaign no.
    if season is None:
        campaign = Season(EPIC)
        if hasattr(campaign, '__len__'):
            raise AttributeError(
                "Please choose a campaign/season for this target: %s."
                % campaign)
    else:
        campaign = season
    epics, kepmags, channels, short_cadence = np.array(GetK2Stars()[
                                                       campaign]).T
    short_cadence = np.array(short_cadence, dtype=bool)
    epics = np.array(epics, dtype=int)
    c = GetNeighboringChannels(Channel(EPIC, campaign=season))

    # Manage kwargs
    if aperture_name is None:
        aperture_name = 'k2sff_15'
    if mag_range is None:
        mag_lo = -np.inf
        mag_hi = np.inf
    else:
        mag_lo = mag_range[0]
        mag_hi = mag_range[1]
        # K2-specific tweak. The short cadence stars are preferentially
        # really bright ones, so we won't get many neighbors if we
        # stick to the default magnitude range! I'm
        # therefore enforcing a lower magnitude cut-off of 8.
        if cadence == 'sc':
            mag_lo = 8.
    if cdpp_range is None:
        cdpp_lo = -np.inf
        cdpp_hi = np.inf
    else:
        cdpp_lo = cdpp_range[0]
        cdpp_hi = cdpp_range[1]
    targets = []

    # First look for nearby targets, then relax the constraint
    # If still no targets, widen magnitude range
    for n in range(3):

        if n == 0:
            nearby = True
        elif n == 1:
            nearby = False
        elif n == 2:
            mag_lo -= 1
            mag_hi += 1

        # Loop over all stars
        for star, kp, channel, sc in zip(epics, kepmags, channels, short_cadence):

            # Preliminary vetting
            if not (((channel in c) if nearby else True) and (kp < mag_hi) \
                    and (kp > mag_lo) and (sc if cadence == 'sc' else True)):
                continue

            # Reject if self or if already in list
            if (star == EPIC) or (star in targets):
                continue

            # Ensure raw light curve file exists
            if not os.path.exists(
                    os.path.join(TargetDirectory(star, campaign), 'data.npz')):
                continue

            # Ensure crowding is OK. This is quite conservative, as we
            # need to prevent potential astrophysical false positive
            # contamination from crowded planet-hosting neighbors when
            # doing neighboring PLD.
            contam = False
            data = np.load(os.path.join(
                TargetDirectory(star, campaign), 'data.npz'))
            aperture = data['apertures'][()][aperture_name]

            # Check that the aperture exists!
            if aperture is None:
                continue

            fpix = data['fpix']
            for source in data['nearby'][()]:
                # Ignore self
                if source['ID'] == star:
                    continue
                # Ignore really dim stars
                if source['mag'] < kp - 5:
                    continue
                # Compute source position
                x = int(np.round(source['x'] - source['x0']))
                y = int(np.round(source['y'] - source['y0']))
                # If the source is within two pixels of the edge
                # of the target aperture, reject the target
                for j in [x - 2, x - 1, x, x + 1, x + 2]:
                    if j < 0:
                        # Outside the postage stamp
                        continue
                    for i in [y - 2, y - 1, y, y + 1, y + 2]:
                        if i < 0:
                            # Outside the postage stamp
                            continue
                        try:
                            if aperture[i][j]:
                                # Oh-oh!
                                contam = True
                        except IndexError:
                            # Out of bounds... carry on!
                            pass
            if contam:
                continue

            # HACK: This happens for K2SFF M67 targets in C05.
            # Let's skip them
            if aperture.shape != fpix.shape[1:]:
                continue

            # Reject if the model is not present
            if model is not None:
                if not os.path.exists(os.path.join(
                        TargetDirectory(star, campaign), model + '.npz')):
                    continue

                # Reject if CDPP out of range
                if cdpp_range is not None:
                    cdpp = np.load(os.path.join(TargetDirectory(
                        star, campaign), model + '.npz'))['cdpp']
                    if (cdpp > cdpp_hi) or (cdpp < cdpp_lo):
                        continue

            # Passed all the tests!
            targets.append(star)

            # Do we have enough? If so, return
            if len(targets) == neighbors:
                random.shuffle(targets)
                return targets

    # If we get to this point, we didn't find enough neighbors...
    # Return what we have anyway.
    return targets