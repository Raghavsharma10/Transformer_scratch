def search_generator(self, query, method="lucene",
                         pagination=10, access_token=None):
        """Search the ORCID database with a generator.

        The generator will yield every result.

        Parameters
        ----------
        :param query: string
            Query in line with the chosen method.
        :param method: string
            One of 'lucene', 'edismax', 'dismax'
        :param pagination: integer
            How many papers should be fetched with the request.
        :param access_token: string
            If obtained before, the access token to use to pass through
            authorization. Note that if this argument is not provided,
            the function will take more time.

        Yields
        -------
        :yields: dict
            Single profile from the search results.
        """
        if access_token is None:
            access_token = self. \
                get_search_token_from_orcid()

        headers = {'Accept': 'application/orcid+json',
                   'Authorization': 'Bearer %s' % access_token}

        index = 0

        while True:
            paginated_result = self._search(query, method, index, pagination,
                                            headers, self._endpoint)
            if not paginated_result['result']:
                return

            for result in paginated_result['result']:
                yield result
            index += pagination