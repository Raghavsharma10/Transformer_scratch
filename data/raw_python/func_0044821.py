def humanize_time(secs):
    """convert second in to hh:mm:ss format
    """
    if secs is None:
        return '--'

    if secs < 1:
        return "{:.2f}ms".format(secs*1000)
    elif secs < 10:
        return "{:.2f}s".format(secs)
    else:
        mins, secs = divmod(secs, 60)
        hours, mins = divmod(mins, 60)
        return '{:02d}:{:02d}:{:02d}'.format(int(hours), int(mins), int(secs))