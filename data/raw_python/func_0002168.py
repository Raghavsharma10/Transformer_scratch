def get_query(self, query):
        """Make a GET request, including a query, to the endpoint.

        The path of the request is to the base URL assigned to the endpoint.

        Parameters
        ----------
        query : DataQuery
            The query to pass when making the request

        Returns
        -------
        resp : requests.Response
            The server's response to the request

        See Also
        --------
        get_path, get

        """
        url = self._base[:-1] if self._base[-1] == '/' else self._base
        return self.get(url, query)