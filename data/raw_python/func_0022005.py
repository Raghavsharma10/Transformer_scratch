def get(self, url, content_type=JSON_CONTENT_TYPE, **kwargs):
        """
        Send GET request and check response.

        :param str method: The HTTP method to use.
        :param str url: The URL to make the request to.

        :raises txacme.client.ServerError: If server response body carries HTTP
            Problem (draft-ietf-appsawg-http-problem-00).
        :raises acme.errors.ClientError: In case of other protocol errors.

        :return: Deferred firing with the checked HTTP response.
        """
        with LOG_JWS_GET().context():
            return (
                DeferredContext(self._send_request(u'GET', url, **kwargs))
                .addCallback(self._check_response, content_type=content_type)
                .addActionFinish())