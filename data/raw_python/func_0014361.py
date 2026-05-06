def _http_request(self, method, url, request_kwargs=None):
        """
        Performs the requested HTTP request.
        """

        kwargs = request_kwargs if request_kwargs is not None else {}

        headers = self._request_headers()
        headers.update(self.additional_headers)
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        kwargs['headers'] = headers

        if self._has_proxy():
            kwargs['proxies'] = self._proxy_parameters()

        request_url = self._url(
            url,
            file_upload=kwargs.pop('file_upload', False)
        )

        request_method = getattr(requests, method)
        response = request_method(request_url, **kwargs)

        if response.status_code == 429:
            raise RateLimitExceededError(response)

        return response