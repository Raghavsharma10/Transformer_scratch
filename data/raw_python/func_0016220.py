def _delete(self, *args, **kwargs):
        """Wrapper around Requests for DELETE requests

        Returns:
            Response:
                A Requests Response object
        """

        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout

        req = self.session.delete(*args, **kwargs)
        return req