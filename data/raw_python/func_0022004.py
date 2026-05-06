def head(self, url, *args, **kwargs):
        """
        Send HEAD request without checking the response.

        Note that ``_check_response`` is not called, as there will be no
        response body to check.

        :param str url: The URL to make the request to.
        """
        with LOG_JWS_HEAD().context():
            return DeferredContext(
                self._send_request(u'HEAD', url, *args, **kwargs)
                ).addActionFinish()