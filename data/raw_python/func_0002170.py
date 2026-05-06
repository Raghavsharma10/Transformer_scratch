def get(self, path, params=None):
        """Make a GET request, optionally including a parameters, to a path.

        The path of the request is the full URL.

        Parameters
        ----------
        path : str
            The URL to request
        params : DataQuery, optional
            The query to pass when making the request

        Returns
        -------
        resp : requests.Response
            The server's response to the request

        Raises
        ------
        HTTPError
            If the server returns anything other than a 200 (OK) code

        See Also
        --------
        get_query, get

        """
        resp = self._session.get(path, params=params)
        if resp.status_code != 200:
            if resp.headers.get('Content-Type', '').startswith('text/html'):
                text = resp.reason
            else:
                text = resp.text
            raise requests.HTTPError('Error accessing {0}\n'
                                     'Server Error ({1:d}: {2})'.format(resp.request.url,
                                                                        resp.status_code,
                                                                        text))
        return resp