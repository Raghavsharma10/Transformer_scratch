def _get_config_instance(group_or_term, session, **kwargs):
    """ Finds appropriate config instance and returns it.

    Args:
        group_or_term (Group or Term):
        session (Sqlalchemy session):
        kwargs (dict): kwargs to pass to get_or_create.

    Returns:
        tuple of (Config, bool):
    """
    path = group_or_term._get_path()
    cached = group_or_term._top._cached_configs.get(path)
    if cached:
        config = cached
        created = False
    else:
        # does not exist or not yet cached
        config, created = get_or_create(session, Config, **kwargs)
    return config, created