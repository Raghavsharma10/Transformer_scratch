def iter_following(username, number=-1, etag=None):
    """List the people ``username`` follows.

    :param str username: (required), login of the user
    :param int number: (optional), number of users being followed by username
        to return. Default: -1, return all of them
    :param str etag: (optional), ETag from a previous request to the same
        endpoint
    :returns: generator of :class:`User <github3.users.User>`

    """
    return gh.iter_following(username, number, etag) if username else []