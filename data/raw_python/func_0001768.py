def created(cls, data=None):
        """Shortcut API for HTTP 201 `Created` response.

        Args:
            data (object): Response key/value data.

        Returns:
            WSResponse Instance.
        """
        if cls.expose_status:  # pragma: no cover
            cls.response.content_type = 'application/json'
            cls.response._status_line = '201 Created'

        return cls(201, data=data).to_json