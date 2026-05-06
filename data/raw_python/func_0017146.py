def iter_repo_issues(owner, repository, milestone=None, state=None,
                     assignee=None, mentioned=None, labels=None, sort=None,
                     direction=None, since=None, number=-1, etag=None):
    """List issues on owner/repository. Only owner and repository are
    required.

    .. versionchanged:: 0.9.0

        - The ``state`` parameter now accepts 'all' in addition to 'open'
          and 'closed'.

    :param str owner: login of the owner of the repository
    :param str repository: name of the repository
    :param int milestone: None, '*', or ID of milestone
    :param str state: accepted values: ('all', 'open', 'closed')
        api-default: 'open'
    :param str assignee: '*' or login of the user
    :param str mentioned: login of the user
    :param str labels: comma-separated list of label names, e.g.,
        'bug,ui,@high'
    :param str sort: accepted values: ('created', 'updated', 'comments')
        api-default: created
    :param str direction: accepted values: ('asc', 'desc')
        api-default: desc
    :param since: (optional), Only issues after this date will
        be returned. This can be a `datetime` or an ISO8601 formatted
        date string, e.g., 2012-05-20T23:10:27Z
    :type since: datetime or string
    :param int number: (optional), number of issues to return.
        Default: -1 returns all issues
    :param str etag: (optional), ETag from a previous request to the same
        endpoint
    :returns: generator of :class:`Issue <github3.issues.Issue>`\ s

    """
    if owner and repository:
        return gh.iter_repo_issues(owner, repository, milestone, state,
                                   assignee, mentioned, labels, sort,
                                   direction, since, number, etag)
    return iter([])