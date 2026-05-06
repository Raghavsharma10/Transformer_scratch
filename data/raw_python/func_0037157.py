def Citation(ivorn, cite_type):
    """
    Deprecated alias of :func:`.EventIvorn`
    """
    import warnings
    warnings.warn(
        """
        `Citation` class has been renamed `EventIvorn` to reflect naming
        conventions in the VOEvent standard.
        As such this class name is a deprecated alias and may be removed in a
        future release.
        """,
        FutureWarning)
    return EventIvorn(ivorn, cite_type)