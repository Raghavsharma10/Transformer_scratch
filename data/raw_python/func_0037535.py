def pull_isotime(voevent, index=0):
    """
    Deprecated alias of :func:`.get_event_time_as_utc`
    """
    import warnings
    warnings.warn(
        """
        The function `pull_isotime` has been renamed to
        `get_event_time_as_utc`. This alias is preserved for backwards
        compatibility, and may be removed in a future release.
        """,
        FutureWarning)
    return get_event_time_as_utc(voevent, index)