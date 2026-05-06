def errorDist(scale, measExpTime, n_events_in_expTime,
              event_duration, std,
              points_per_time=100, n_repetitions=300):
    '''
    TODO
    '''
    ntimes = 10
    s1 = measExpTime * scale * 10
    # exp. time factor 1/16-->16:
    p2 = np.logspace(-4, 4, 18, base=2)

    t = np.linspace(0, s1, ntimes * points_per_time * s1)

    err = None
    for rr in range(n_repetitions):

        f = _flux(t, n_events_in_expTime, event_duration, std)

        e = np.array([_capture(f, t, measExpTime, pp) for pp in p2])
        if err is None:
            err = e
        else:
            err += e
    err /= (rr + 1)
    
    # normalize, so that error==1 at 1:
    try:
        fac = findXAt(err, p2, 1)
    except:
        fac = 1

    err /= fac
    return p2, err, t, f