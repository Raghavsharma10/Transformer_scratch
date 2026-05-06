def not_modified(cls, errors=None):
        """Shortcut API for HTTP 304 `Not Modified` response.

        Args:
            errors (list): Response key/value data.

        Returns:
            WSResponse Instance.
        """
        if cls.expose_status:  # pragma: no cover
            cls.response.content_type = 'application/json'
            cls.response._status_line = '304 Not Modified'

        return cls(304, None, errors).to_json