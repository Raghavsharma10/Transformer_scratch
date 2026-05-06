def execute_request(self, action, path, data=None, headers_ext=None):
        """Generate request to WebDAV server for specified action and path and execute it.

        :param action: the action for WebDAV server which should be executed.
        :param path: the path to resource for action
        :param data: (optional) Dictionary or list of tuples ``[(key, value)]`` (will be form-encoded), bytes,
                     or file-like object to send in the body of the :class:`Request`.
        :param headers_ext: (optional) the addition headers list witch should be added to basic HTTP headers for
                            the specified action.
        :return: HTTP response of request.
        """
        response = self.session.request(
            method=Client.requests[action],
            url=self.get_url(path),
            auth=self.webdav.auth,
            headers=self.get_headers(action, headers_ext),
            timeout=self.timeout,
            data=data
        )
        if response.status_code == 507:
            raise NotEnoughSpace()
        if 499 < response.status_code < 600:
            raise ServerException(url=self.get_url(path), code=response.status_code, message=response.content)
        if response.status_code >= 400:
            raise ResponseErrorCode(url=self.get_url(path), code=response.status_code, message=response.content)
        return response