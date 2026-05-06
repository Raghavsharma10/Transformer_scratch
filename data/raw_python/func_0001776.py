def service_unavailable(cls, errors=None):
        """Shortcut API for HTTP 503 `Service Unavailable` response.

        Args:
            errors (list): Response key/value data.

        Returns:
            WSResponse Instance.
        """
        if cls.expose_status:  # pragma: no cover
            cls.response.content_type = 'application/json'
            cls.response._status_line = '503 Service Unavailable'

        return cls(503, None, errors).to_json