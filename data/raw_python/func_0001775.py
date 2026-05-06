def not_implemented(cls, errors=None):
        """Shortcut API for HTTP 501 `Not Implemented` response.

        Args:
            errors (list): Response key/value data.

        Returns:
            WSResponse Instance.
        """
        if cls.expose_status:  # pragma: no cover
            cls.response.content_type = 'application/json'
            cls.response._status_line = '501 Not Implemented'

        return cls(501, None, errors).to_json