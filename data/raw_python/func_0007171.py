def is_older_than(before, seconds):
    """Return True if before is older than seconds."""
    if isinstance(before, six.string_types):
        before = parse_strtime(before).replace(tzinfo=None)
    else:
        before = before.replace(tzinfo=None)

    return utcnow() - before > datetime.timedelta(seconds=seconds)