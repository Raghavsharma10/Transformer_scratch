def search_repositories(query, sort=None, order=None, per_page=None,
                        text_match=False, number=-1, etag=None):
    """Find repositories via various criteria.

    .. warning::

        You will only be able to make 5 calls with this or other search
        functions. To raise the rate-limit on this set of endpoints, create an
        authenticated :class:`GitHub <github3.github.GitHub>` Session with
        ``login``.

    The query can contain any combination of the following supported
    qualifers:

    - ``in`` Qualifies which fields are searched. With this qualifier you
      can restrict the search to just the repository name, description,
      readme, or any combination of these.
    - ``size`` Finds repositories that match a certain size (in
      kilobytes).
    - ``forks`` Filters repositories based on the number of forks, and/or
      whether forked repositories should be included in the results at
      all.
    - ``created`` or ``pushed`` Filters repositories based on times of
      creation, or when they were last updated. Format: ``YYYY-MM-DD``.
      Examples: ``created:<2011``, ``pushed:<2013-02``,
      ``pushed:>=2013-03-06``
    - ``user`` or ``repo`` Limits searches to a specific user or
      repository.
    - ``language`` Searches repositories based on the language they're
      written in.
    - ``stars`` Searches repositories based on the number of stars.

    For more information about these qualifiers, see: http://git.io/4Z8AkA

    :param str query: (required), a valid query as described above, e.g.,
        ``tetris language:assembly``
    :param str sort: (optional), how the results should be sorted;
        options: ``stars``, ``forks``, ``updated``; default: best match
    :param str order: (optional), the direction of the sorted results,
        options: ``asc``, ``desc``; default: ``desc``
    :param int per_page: (optional)
    :param bool text_match: (optional), if True, return matching search
        terms. See http://git.io/4ct1eQ for more information
    :param int number: (optional), number of repositories to return.
        Default: -1, returns all available repositories
    :param str etag: (optional), previous ETag header value
    :return: generator of :class:`Repository <github3.repos.Repository>`
    """
    return gh.search_repositories(query, sort, order, per_page, text_match,
                                  number, etag)