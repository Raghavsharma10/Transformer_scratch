def iter_followers(username, number=-1, etag=None):
    """List the followers of ``username``.

    :param str username: (required), login of the person to list the followers
        of
    :param int number: (optional), number of followers to return, Default: -1,
        return all of them
    :param str etag: (optional), ETag from a previous request to the same
        endpoint
    :returns: generator of :class:`User <github3.users.User>`

    """
    return gh.iter_followers(username, number, etag) if username else []