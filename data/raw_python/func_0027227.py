def update(self, request, *args, **kwargs):
        """
        To update service settings, issue a **PUT** or **PATCH** to */api/service-settings/<uuid>/* as a customer owner.
        You are allowed to change name and credentials only.

        Example of a request:

        .. code-block:: http

            PATCH /api/service-settings/9079705c17d64e6aa0af2e619b0e0702/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "username": "admin",
                "password": "new_secret"
            }
        """
        return super(ServiceSettingsViewSet, self).update(request, *args, **kwargs)