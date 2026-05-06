def forbidden(cls, errors=None):
        """Shortcut API for HTTP 403 `Forbidden` response.

        Args:
            errors (list): Response key/value data.

        Returns:
            WSResponse Instance.
        """
        if cls.expose_status:  # pragma: no cover
            cls.response.content_type = 'application/json'
            cls.response._status_line = '403 Forbidden'

        return cls(403, errors=errors).to_json