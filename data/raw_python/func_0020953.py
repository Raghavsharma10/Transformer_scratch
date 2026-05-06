def search(self, query, method="lucene", start=None,
               rows=None, access_token=None):
        """Search the ORCID database.

        Parameters
        ----------
        :param query: string
            Query in line with the chosen method.
        :param method: string
            One of 'lucene', 'edismax', 'dismax'
        :param start: string
            Index of the first record requested. Use for pagination.
        :param rows: string
            Number of records requested. Use for pagination.
        :param access_token: string
            If obtained before, the access token to use to pass through
            authorization. Note that if this argument is not provided,
            the function will take more time.

        Returns
        -------
        :returns: dict
            Search result with error description available. The results can
            be obtained by accessing key 'result'. To get the number
            of all results, access the key 'num-found'.
        """
        if access_token is None:
            access_token = self. \
                get_search_token_from_orcid()

        headers = {'Accept': 'application/orcid+json',
                   'Authorization': 'Bearer %s' % access_token}

        return self._search(query, method, start, rows, headers,
                            self._endpoint)