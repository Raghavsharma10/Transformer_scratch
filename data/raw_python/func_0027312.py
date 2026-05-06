def create(self, request, *args, **kwargs):
        """
        Run **POST** against */api/alerts/* to create or update alert. If alert with posted scope and
        alert_type already exists - it will be updated. Only users with staff privileges can create alerts.

        Request example:

        .. code-block:: javascript

            POST /api/alerts/
            Accept: application/json
            Content-Type: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "scope": "http://testserver/api/projects/b9e8a102b5ff4469b9ac03253fae4b95/",
                "message": "message#1",
                "alert_type": "first_alert",
                "severity": "Debug"
            }
        """
        return super(AlertViewSet, self).create(request, *args, **kwargs)