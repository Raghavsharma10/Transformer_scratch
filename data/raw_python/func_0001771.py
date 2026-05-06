def unauthorized(cls, errors=None):
        """Shortcut API for HTTP 401 `Unauthorized` response.

        Args:
            errors (list): Response key/value data.

        Returns:
            WSResponse Instance.
        """
        if cls.expose_status:  # pragma: no cover
            cls.response.content_type = 'application/json'
            cls.response._status_line = '401 Unauthorized'

        return cls(401, errors=errors).to_json