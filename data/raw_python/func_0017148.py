def iter_user_repos(login, type=None, sort=None, direction=None, number=-1,
                    etag=None):
    """List public repositories for the specified ``login``.

    .. versionadded:: 0.6

    .. note:: This replaces github3.iter_repos

    :param str login: (required)
    :param str type: (optional), accepted values:
        ('all', 'owner', 'member')
        API default: 'all'
    :param str sort: (optional), accepted values:
        ('created', 'updated', 'pushed', 'full_name')
        API default: 'created'
    :param str direction: (optional), accepted values:
        ('asc', 'desc'), API default: 'asc' when using 'full_name',
        'desc' otherwise
    :param int number: (optional), number of repositories to return.
        Default: -1 returns all repositories
    :param str etag: (optional), ETag from a previous request to the same
        endpoint
    :returns: generator of :class:`Repository <github3.repos.Repository>`
        objects

    """
    if login:
        return gh.iter_user_repos(login, type, sort, direction, number, etag)
    return iter([])