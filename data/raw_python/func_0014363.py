def _http_post(self, url, data, **kwargs):
        """
        Performs the HTTP POST request.
        """

        if not kwargs.get('file_upload', False):
            data = json.dumps(data)

        kwargs.update({'data': data})

        return self._http_request('post', url, kwargs)