def _http_put(self, url, data, **kwargs):
        """
        Performs the HTTP PUT request.
        """

        kwargs.update({'data': json.dumps(data)})

        return self._http_request('put', url, kwargs)