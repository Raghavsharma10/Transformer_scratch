def request(self, url, *,
                method='GET', headers=None, data=None, result_callback=None):
        """Perform request.

        :param str url: request URL.
        :param str method: request method.
        :param dict headers: request headers.
        :param object data: request data.
        :param object -> object result_callback: result callback.

        :rtype: dict
        :raise: APIError
        """

        url = self._make_full_url(url)

        self._log.debug('Performing %s request to %s', method, url)
        return self._request(url, method=method, headers=headers, data=data,
                             result_callback=result_callback)