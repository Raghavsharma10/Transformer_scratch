def _put(self, *args, **kwargs):
        """Wrapper around Requests for PUT requests

        Returns:
            Response:
                A Requests Response object
        """

        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout

        req = self.session.put(*args, **kwargs)
        return req