def search_requests(self, query=None, params=None, callback=None, mine_ids=None):
        """Mine Archive.org search results.

        :param query: The Archive.org search query to yield results for.
                      Refer to https://archive.org/advancedsearch.php#raw
                      for help formatting your query.
        :type query: str

        :param params: The URL parameters to send with each request sent
                       to the Archive.org Advancedsearch Api.
        :type params: dict
        """
        # If mining ids, devote half the workers to search and half to item mining.
        if mine_ids:
            self.max_tasks = self.max_tasks/2
        # When mining id's, the only field we need returned is "identifier".
        if mine_ids and params:
            params = dict((k, v) for k, v in params.items() if 'fl' not in k)
            params['fl[]'] = 'identifier'

        # Make sure "identifier" is always returned in search results.
        fields = [k for k in params if 'fl' in k]
        if (len(fields) == 1) and (not any('identifier' == params[k] for k in params)):
            # Make sure to not overwrite the existing fl[] key.
            i = 0
            while params.get('fl[{}]'.format(i)):
                i += 1
            params['fl[{}]'.format(i)] = 'identifier'

        search_params = self.get_search_params(query, params)
        url = make_url('/advancedsearch.php', self.protocol, self.hosts)

        search_info = self.get_search_info(search_params)
        total_results = search_info.get('response', {}).get('numFound', 0)
        total_pages = (int(total_results/search_params['rows']) + 1)

        for page in range(1, (total_pages + 1)):
            params = deepcopy(search_params)
            params['page'] = page
            if not callback and mine_ids:
                callback = self._handle_search_results
            req = MineRequest('GET', url, self.access,
                              callback=callback,
                              max_retries=self.max_retries,
                              debug=self.debug,
                              params=params,
                              connector=self.connector)
            yield req