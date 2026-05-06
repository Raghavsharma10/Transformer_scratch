def _post(self, *args, **kwargs):
        """Wrapper around Requests for POST requests

        Returns:
            Response:
                A Requests Response object
        """

        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout

        req = self.session.post(*args, **kwargs)
        return req