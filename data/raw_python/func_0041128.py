def format_duration(secs):
    """
    Format a duration in seconds as minutes and seconds.
    """
    secs = int(secs)

    if abs(secs) > 60:
        mins = abs(secs) / 60
        secs = abs(secs) - (mins * 60)

        return '%s%im %02is' % ('-' if secs < 0 else '', mins, secs)

    return '%is' % secs