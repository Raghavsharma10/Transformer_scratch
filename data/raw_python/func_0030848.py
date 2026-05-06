def _prepare_request(self, url, method, headers, data):
        """Prepare HTTP request.

        :param str url: request URL.
        :param str method: request method.
        :param dict headers: request headers.
        :param object data: JSON-encodable object.

        :rtype: httpclient.HTTPRequest

        """

        request = httpclient.HTTPRequest(
            url=url, method=method, headers=headers, body=data,
            connect_timeout=self._connect_timeout,
            request_timeout=self._request_timeout,
            auth_username=self._username, auth_password=self._password,
            client_cert=self._client_cert, client_key=self._client_key,
            ca_certs=self._ca_certs, validate_cert=self._verify_cert)

        return request