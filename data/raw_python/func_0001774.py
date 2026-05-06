def method_not_allowed(cls, errors=None):
        """Shortcut API for HTTP 405 `Method not allowed` response.

        Args:
            errors (list): Response key/value data.

        Returns:
            WSResponse Instance.
        """
        if cls.expose_status:  # pragma: no cover
            cls.response.content_type = 'application/json'
            cls.response._status_line = '405 Method Not Allowed'

        return cls(405, None, errors).to_json