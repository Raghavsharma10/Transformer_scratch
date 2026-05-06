def search_issues(self, query, sort=None, order=None, per_page=None,
                      text_match=False, number=-1, etag=None):
        """Find issues by state and keyword

        The query can contain any combination of the following supported
        qualifers:

        - ``type`` With this qualifier you can restrict the search to issues
          or pull request only.
        - ``in`` Qualifies which fields are searched. With this qualifier you
          can restrict the search to just the title, body, comments, or any
          combination of these.
        - ``author`` Finds issues created by a certain user.
        - ``assignee`` Finds issues that are assigned to a certain user.
        - ``mentions`` Finds issues that mention a certain user.
        - ``commenter`` Finds issues that a certain user commented on.
        - ``involves`` Finds issues that were either created by a certain user,
          assigned to that user, mention that user, or were commented on by
          that user.
        - ``state`` Filter issues based on whether they’re open or closed.
        - ``labels`` Filters issues based on their labels.
        - ``language`` Searches for issues within repositories that match a
          certain language.
        - ``created`` or ``updated`` Filters issues based on times of creation,
          or when they were last updated.
        - ``comments`` Filters issues based on the quantity of comments.
        - ``user`` or ``repo`` Limits searches to a specific user or
          repository.

        For more information about these qualifiers, see: http://git.io/d1oELA

        :param str query: (required), a valid query as described above, e.g.,
            ``windows label:bug``
        :param str sort: (optional), how the results should be sorted;
            options: ``created``, ``comments``, ``updated``;
            default: best match
        :param str order: (optional), the direction of the sorted results,
            options: ``asc``, ``desc``; default: ``desc``
        :param int per_page: (optional)
        :param bool text_match: (optional), if True, return matching search
          terms. See http://git.io/QLQuSQ for more information
        :param int number: (optional), number of issues to return.
            Default: -1, returns all available issues
        :param str etag: (optional), previous ETag header value
        :return: generator of :class:`IssueSearchResult
            <github3.search.IssueSearchResult>`
        """
        params = {'q': query}
        headers = {}

        if sort in ('comments', 'created', 'updated'):
            params['sort'] = sort

        if order in ('asc', 'desc'):
            params['order'] = order

        if text_match:
            headers = {
                'Accept': 'application/vnd.github.v3.full.text-match+json'
                }

        url = self._build_url('search', 'issues')
        return SearchIterator(number, url, IssueSearchResult, self, params,
                              etag, headers)