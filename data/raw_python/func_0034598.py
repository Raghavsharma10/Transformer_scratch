def convert_time_units(t):
    """ Convert time in seconds into reasonable time units. """
    if t == 0:
        return '0 s'
    order = log10(t)
    if -9 < order < -6:
        time_units = 'ns'
        factor = 1000000000
    elif -6 <= order < -3:
        time_units = 'us'
        factor = 1000000
    elif -3 <= order < -1:
        time_units = 'ms'
        factor = 1000.
    elif -1 <= order:
        time_units = 's'
        factor = 1
    return "{:.3f} {}".format(factor * t, time_units)