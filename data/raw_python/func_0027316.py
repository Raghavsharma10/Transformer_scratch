def create(self, request, *args, **kwargs):
        """
        To create new web hook issue **POST** against */api/hooks-web/* as an authenticated user.
        You should specify list of event_types or event_groups.

        Example of a request:

        .. code-block:: http

            POST /api/hooks-web/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "event_types": ["resource_start_succeeded"],
                "event_groups": ["users"],
                "destination_url": "http://example.com/"
            }

        When hook is activated, **POST** request is issued against destination URL with the following data:

        .. code-block:: javascript

            {
                "timestamp": "2015-07-14T12:12:56.000000",
                "message": "Customer ABC LLC has been updated.",
                "type": "customer_update_succeeded",
                "context": {
                    "user_native_name": "Walter Lebrowski",
                    "customer_contact_details": "",
                    "user_username": "Walter",
                    "user_uuid": "1c3323fc4ae44120b57ec40dea1be6e6",
                    "customer_uuid": "4633bbbb0b3a4b91bffc0e18f853de85",
                    "ip_address": "8.8.8.8",
                    "user_full_name": "Walter Lebrowski",
                    "customer_abbreviation": "ABC LLC",
                    "customer_name": "ABC LLC"
                },
                "levelname": "INFO"
            }

        Note that context depends on event type.
        """
        return super(WebHookViewSet, self).create(request, *args, **kwargs)