def protected_branches():
    # type: () -> list[str]
    """ Return branches protected by deletion.

    By default those are master and devel branches as configured in pelconf.

    Returns:
        list[str]: Names of important branches that should not be deleted.
    """
    master = conf.get('git.master_branch', 'master')
    develop = conf.get('git.devel_branch', 'develop')
    return conf.get('git.protected_branches', (master, develop))