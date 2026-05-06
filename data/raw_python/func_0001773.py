def not_found(cls, errors=None):
        """Shortcut API for HTTP 404 `Not found` response.

        Args:
            errors (list): Response key/value data.

        Returns:
            WSResponse Instance.
        """
        if cls.expose_status:  # pragma: no cover
            cls.response.content_type = 'application/json'
            cls.response._status_line = '404 Not Found'

        return cls(404, None, errors).to_json