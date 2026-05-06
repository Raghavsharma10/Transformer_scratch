def _get_xml(self, *args, **kwargs):
        """Wrapper around Requests for GET XML requests

        Returns:
            Response:
                A Requests Response object
        """
        req = self.session_xml.get(*args, **kwargs)
        return req