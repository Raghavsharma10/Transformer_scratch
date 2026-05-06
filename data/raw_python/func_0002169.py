def get_path(self, path, query=None):
        """Make a GET request, optionally including a query, to a relative path.

        The path of the request includes a path on top of the base URL
        assigned to the endpoint.

        Parameters
        ----------
        path : str
            The path to request, relative to the endpoint
        query : DataQuery, optional
            The query to pass when making the request

        Returns
        -------
        resp : requests.Response
            The server's response to the request

        See Also
        --------
        get_query, get, url_path

        """
        return self.get(self.url_path(path), query)