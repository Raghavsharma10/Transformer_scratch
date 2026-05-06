def slice_time(begin, end=None, duration=datetime.timedelta(days=2)):
    """
    :param begin: datetime
    :param end: datetime
    :param duration: timedelta
    :return: a generator for a set of timeslices of the given duration
    """
    duration_ms = int(duration.total_seconds() * 1000)
    previous = int(unix_time(begin) * 1000)
    next = previous + duration_ms
    now_ms = unix_time(datetime.datetime.now())*1000
    end_slice = now_ms if not end else min(now_ms, int(unix_time(end) * 1000))

    while next < end_slice:
        yield TimeSlice(previous, next)
        previous = next
        next += duration_ms
        now_ms = unix_time(datetime.datetime.now())*1000
        end_slice = now_ms if not end else min(now_ms, int(unix_time(end) * 1000))
    yield TimeSlice(previous, end_slice)