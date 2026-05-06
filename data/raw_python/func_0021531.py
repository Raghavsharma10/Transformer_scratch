def request(self, method, path,
                params=None, headers=None, cookies=None, data=None, json=None, allow_redirects=None, timeout=None):
        """
        Prepares and sends an HTTP request. Returns the HTTPResponse object.

        :param method: str
        :param path: str
        :return: response
        :rtype: HTTPResponse
        """
        headers = headers or {}
        timeout = timeout if timeout is not None else self._timeout
        allow_redirects = allow_redirects if allow_redirects is not None else self._allow_redirects

        if self._keep_alive and self.__session is None:
            self.__session = requests.Session()

        if self.__session is not None and not self._use_cookies:
            self.__session.cookies.clear()

        address = self._bake_address(path)
        req_headers = copy.deepcopy(self._additional_headers)
        req_headers.update(headers)

        response = http.request(method, address, session=self.__session,
                                params=params, headers=headers, cookies=cookies, data=data, json=json,
                                allow_redirects=allow_redirects, timeout=timeout)
        if self._auto_assert_ok:
            response.assert_ok()
        return response