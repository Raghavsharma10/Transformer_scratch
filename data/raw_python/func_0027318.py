def create(self, request, *args, **kwargs):
        """
        To create new push hook issue **POST** against */api/hooks-push/* as an authenticated user.
        You should specify list of event_types or event_groups.

        Example of a request:

        .. code-block:: http

            POST /api/hooks-push/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "event_types": ["resource_start_succeeded"],
                "event_groups": ["users"],
                "type": "Android"
            }

        You may temporarily disable hook without deleting it by issuing following **PATCH** request against hook URL:

        .. code-block:: javascript

            {
                "is_active": "false"
            }
        """
        return super(PushHookViewSet, self).create(request, *args, **kwargs)