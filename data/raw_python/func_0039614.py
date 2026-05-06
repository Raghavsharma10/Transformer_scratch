def to_ut1unix(time: Union[str, datetime, float, np.ndarray]) -> np.ndarray:
    """
    converts time inputs to UT1 seconds since Unix epoch
    """
    # keep this order
    time = totime(time)

    if isinstance(time, (float, int)):
        return time

    if isinstance(time, (tuple, list, np.ndarray)):
        assert isinstance(time[0], datetime), f'expected datetime, not {type(time[0])}'
        return np.array(list(map(dt2ut1, time)))
    else:
        assert isinstance(time, datetime)
        return dt2ut1(time)