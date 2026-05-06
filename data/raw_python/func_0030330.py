def stop_if_mostly_diverging(errdata):
    """This is an example stop condition that asks Relay to quit if
    the error difference between consecutive samples is increasing more than
    half of the time.

    It's quite sensitive and designed for the demo, so you probably shouldn't
    use this is a production setting
    """
    n_increases = sum([
        abs(y) - abs(x) > 0 for x, y in zip(errdata, errdata[1:])])
    if len(errdata) * 0.5 < n_increases:
        # most of the time, the next sample is worse than the previous sample
        # relay is not healthy
        return 0
    else:
        # most of the time, the next sample is better than the previous sample
        # realy is in a healthy state
        return -1