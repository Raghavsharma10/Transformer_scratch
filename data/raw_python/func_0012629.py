def saccade_detection(samplemat, Hz=200, threshold=30,
                      acc_thresh=2000, min_duration=21, min_movement=.35,
                      ignore_blinks=False):
    '''
    Detect saccades in a stream of gaze location samples.

    Coordinates in samplemat are assumed to be in degrees.

    Saccades are detect by a velocity/acceleration threshold approach.
    A saccade starts when a) the velocity is above threshold, b) the
    acceleration is above acc_thresh at least once during the interval
    defined by the velocity threshold, c) the saccade lasts at least min_duration
    ms and d) the distance between saccade start and enpoint is at least
    min_movement degrees.
    '''
    if ignore_blinks:
        velocity, acceleration = get_velocity(samplemat, float(Hz), blinks=samplemat.blinks)
    else:
        velocity, acceleration = get_velocity(samplemat, float(Hz))

    saccades = (velocity > threshold)
    #print velocity[samplemat.blinks[1:]]
    #print saccades[samplemat.blinks[1:]]

    borders = np.where(np.diff(saccades.astype(int)))[0] + 1
    if velocity[1] > threshold:
        borders = np.hstack(([0], borders))
    saccade = 0 * np.ones(samplemat.x.shape)

    # Only count saccades when acceleration also surpasses threshold
    for i, (start, end) in enumerate(zip(borders[0::2], borders[1::2])):
        if sum(acceleration[start:end] > acc_thresh) >= 1:
            saccade[start:end] = 1

    borders = np.where(np.diff(saccade.astype(int)))[0] + 1
    if saccade[0] == 0:
        borders = np.hstack(([0], borders))
    for i, (start, end) in enumerate(zip(borders[0::2], borders[1::2])):
        if (1000*(end - start) / float(Hz)) < (min_duration):
            saccade[start:end] = 1

    # Delete saccade between fixations that are too close together.
    dists_ok = False
    while not dists_ok:
        dists_ok = True
        num_merges = 0
        for i, (lfixstart, lfixend, start, end, nfixstart, nfixend) in enumerate(zip(
                borders[0::2], borders[1::2],
                borders[1::2], borders[2::2],
                borders[2::2], borders[3::2])):
            lastx = samplemat.x[lfixstart:lfixend].mean()
            lasty = samplemat.y[lfixstart:lfixend].mean()
            nextx = samplemat.x[nfixstart:nfixend].mean()
            nexty = samplemat.y[nfixstart:nfixend].mean()
            if (1000*(lfixend - lfixstart) / float(Hz)) < (min_duration):
                saccade[lfixstart:lfixend] = 1
                continue
            distance = ((nextx - lastx) ** 2 + (nexty - lasty) ** 2) ** .5
            if distance < min_movement:
                num_merges += 1
                dists_ok = False
                saccade[start:end] = 0
        borders = np.where(np.diff(saccade.astype(int)))[0] + 1
        if saccade[0] == 0:
            borders = np.hstack(([0], borders))
    return saccade.astype(bool)