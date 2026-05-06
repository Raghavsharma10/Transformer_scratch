def search_users(self, query, sort=None, order=None, per_page=None,
                     text_match=False, number=-1, etag=None):
        """Find users via the Search API.

        The query can contain any combination of the following supported
        qualifers:


        - ``type`` With this qualifier you can restrict the search to just
          personal accounts or just organization accounts.
        - ``in`` Qualifies which fields are searched. With this qualifier you
          can restrict the search to just the username, public email, full
          name, or any combination of these.
        - ``repos`` Filters users based on the number of repositories they
          have.
        - ``location`` Filter users by the location indicated in their
          profile.
        - ``language`` Search for users that have repositories that match a
          certain language.
        - ``created`` Filter users based on when they joined.
        - ``followers`` Filter users based on the number of followers they
          have.

        For more information about these qualifiers see: http://git.io/wjVYJw

        :param str query: (required), a valid query as described above, e.g.,
            ``tom repos:>42 followers:>1000``
        :param str sort: (optional), how the results should be sorted;
            options: ``followers``, ``repositories``, or ``joined``; default:
            best match
        :param str order: (optional), the direction of the sorted results,
            options: ``asc``, ``desc``; default: ``desc``
        :param int per_page: (optional)
        :param bool text_match: (optional), if True, return matching search
            terms. See http://git.io/_V1zRwa for more information
        :param int number: (optional), number of search results to return;
            Default: -1 returns all available
        :param str etag: (optional), ETag header value of the last request.
        :return: generator of :class:`UserSearchResult
            <github3.search.UserSearchResult>`
        """
        params = {'q': query}
        headers = {}

        if sort in ('followers', 'repositories', 'joined'):
            params['sort'] = sort

        if order in ('asc', 'desc'):
            params['order'] = order

        if text_match:
            headers = {
                'Accept': 'application/vnd.github.v3.full.text-match+json'
                }

        url = self._build_url('search', 'users')
        return SearchIterator(number, url, UserSearchResult, self, params,
                              etag, headers)