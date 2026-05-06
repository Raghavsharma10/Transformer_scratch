def get_next_version() -> str:
    """
    Returns: next version for this Git repository
    """
    LOGGER.info('computing next version')
    should_be_alpha = bool(CTX.repo.get_current_branch() != 'master')
    LOGGER.info('alpha: %s', should_be_alpha)
    calver = _get_calver()
    LOGGER.info('current calver: %s', calver)
    calver_tags = _get_current_calver_tags(calver)
    LOGGER.info('found %s matching tags for this calver', len(calver_tags))
    next_stable_version = _next_stable_version(calver, calver_tags)
    LOGGER.info('next stable version: %s', next_stable_version)
    if should_be_alpha:
        return _next_alpha_version(next_stable_version, calver_tags)

    return next_stable_version