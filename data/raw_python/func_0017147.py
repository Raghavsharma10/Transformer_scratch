def iter_orgs(username, number=-1, etag=None):
    """List the organizations associated with ``username``.

    :param str username: (required), login of the user
    :param int number: (optional), number of orgs to return. Default: -1,
        return all of the issues
    :param str etag: (optional), ETag from a previous request to the same
        endpoint
    :returns: generator of
        :class:`Organization <github3.orgs.Organization>`

    """
    return gh.iter_orgs(username, number, etag) if username else []