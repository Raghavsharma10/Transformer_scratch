def pull_astro_coords(voevent, index=0):
    """
    Deprecated alias of :func:`.get_event_position`
    """
    import warnings
    warnings.warn(
        """
        The function `pull_astro_coords` has been renamed to
        `get_event_position`. This alias is preserved for backwards
        compatibility, and may be removed in a future release.
        """,
        FutureWarning)
    return get_event_position(voevent, index)