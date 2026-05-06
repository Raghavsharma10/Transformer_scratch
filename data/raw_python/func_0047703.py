def fmt_bytes(bytes, precision=2):
    """Reduce a large number of `bytes` down to a humanised SI 
    equivalent and return the result as a string with trailing unit 
    abbreviation.

    """
    UNITS = ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB']

    if bytes == 0:
        return '0 bytes'

    log = math.floor(math.log(bytes, 1000))

    return "%.*f %s" % (precision,
                       bytes / math.pow(1000, log),
                       UNITS[int(log)])