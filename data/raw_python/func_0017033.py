def search_code(self, query, sort=None, order=None, per_page=None,
                    text_match=False, number=-1, etag=None):
        """Find code via the code search API.

        The query can contain any combination of the following supported
        qualifiers:

        - ``in`` Qualifies which fields are searched. With this qualifier you
          can restrict the search to just the file contents, the file path, or
          both.
        - ``language`` Searches code based on the language it’s written in.
        - ``fork`` Specifies that code from forked repositories should be
          searched.  Repository forks will not be searchable unless the fork
          has more stars than the parent repository.
        - ``size`` Finds files that match a certain size (in bytes).
        - ``path`` Specifies the path that the resulting file must be at.
        - ``extension`` Matches files with a certain extension.
        - ``user`` or ``repo`` Limits searches to a specific user or
          repository.

        For more information about these qualifiers, see: http://git.io/-DvAuA

        :param str query: (required), a valid query as described above, e.g.,
            ``addClass in:file language:js repo:jquery/jquery``
        :param str sort: (optional), how the results should be sorted;
            option(s): ``indexed``; default: best match
        :param str order: (optional), the direction of the sorted results,
            options: ``asc``, ``desc``; default: ``desc``
        :param int per_page: (optional)
        :param bool text_match: (optional), if True, return matching search
            terms. See http://git.io/iRmJxg for more information
        :param int number: (optional), number of repositories to return.
            Default: -1, returns all available repositories
        :param str etag: (optional), previous ETag header value
        :return: generator of :class:`CodeSearchResult
            <github3.search.CodeSearchResult>`
        """
        params = {'q': query}
        headers = {}

        if sort == 'indexed':
            params['sort'] = sort

        if sort and order in ('asc', 'desc'):
            params['order'] = order

        if text_match:
            headers = {
                'Accept': 'application/vnd.github.v3.full.text-match+json'
                }

        url = self._build_url('search', 'code')
        return SearchIterator(number, url, CodeSearchResult, self, params,
                              etag, headers)