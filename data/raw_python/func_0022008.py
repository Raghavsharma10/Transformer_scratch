def _post(self, url, obj, content_type, **kwargs):
        """
        POST an object and check the response.

        :param str url: The URL to request.
        :param ~josepy.interfaces.JSONDeSerializable obj: The serializable
            payload of the request.
        :param bytes content_type: The expected content type of the response.

        :raises txacme.client.ServerError: If server response body carries HTTP
            Problem (draft-ietf-appsawg-http-problem-00).
        :raises acme.errors.ClientError: In case of other protocol errors.
        """
        with LOG_JWS_POST().context():
            headers = kwargs.setdefault('headers', Headers())
            headers.setRawHeaders(b'content-type', [JSON_CONTENT_TYPE])
            return (
                DeferredContext(self._get_nonce(url))
                .addCallback(self._wrap_in_jws, obj)
                .addCallback(
                    lambda data: self._send_request(
                        u'POST', url, data=data, **kwargs))
                .addCallback(self._add_nonce)
                .addCallback(self._check_response, content_type=content_type)
                .addActionFinish())