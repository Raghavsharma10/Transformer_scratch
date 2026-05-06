def timestamp_filename(basename, ext=None):
    """
    Return a string of the form [basename-TIMESTAMP.ext]
    where TIMESTAMP is of the form YYYYMMDD-HHMMSS-MILSEC
    """
    dt = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if ext:
        return '%s-%s.%s' % (basename, dt, ext)
    return '%s-%s' % (basename, dt)