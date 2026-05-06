def _get(self, *args, **kwargs):
        """Wrapper around Requests for GET requests

        Returns:
            Response:
                A Requests Response object
        """

        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout

        req = self.session.get(*args, **kwargs)
        return req