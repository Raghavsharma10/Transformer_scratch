def _post_xml(self, *args, **kwargs):
        """Wrapper around Requests for POST requests

        Returns:
            Response:
                A Requests Response object
        """

        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout

        req = self.session_xml.post(*args, **kwargs)
        return req