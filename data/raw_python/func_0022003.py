def _send_request(self, method, url, *args, **kwargs):
        """
        Send HTTP request.

        :param str method: The HTTP method to use.
        :param str url: The URL to make the request to.

        :return: Deferred firing with the HTTP response.
        """
        action = LOG_JWS_REQUEST(url=url)
        with action.context():
            headers = kwargs.setdefault('headers', Headers())
            headers.setRawHeaders(b'user-agent', [self._user_agent])
            kwargs.setdefault('timeout', self.timeout)
            return (
                DeferredContext(
                    self._treq.request(method, url, *args, **kwargs))
                .addCallback(
                    tap(lambda r: action.add_success_fields(
                        code=r.code,
                        content_type=r.headers.getRawHeaders(
                            b'content-type', [None])[0])))
                .addActionFinish())