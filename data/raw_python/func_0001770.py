def bad_request(cls, errors=None):
        """Shortcut API for HTTP 400 `Bad Request` response.

        Args:
            errors (list): Response key/value data.

        Returns:
            WSResponse Instance.
        """
        if cls.expose_status:  # pragma: no cover
            cls.response.content_type = 'application/json'
            cls.response._status_line = '400 Bad Request'

        return cls(400, errors=errors).to_json