def list(self, request, *args, **kwargs):
        """
        To get a list of events - run **GET** against */api/events/* as authenticated user. Note that a user can
        only see events connected to objects she is allowed to see.

        Sorting is supported in ascending and descending order by specifying a field to an **?o=** parameter. By default
        events are sorted by @timestamp in descending order.

        Run POST against */api/events/* to create an event. Only users with staff privileges can create events.
        New event will be emitted with `custom_notification` event type.
        Request should contain following fields:

        - level: the level of current event. Following levels are supported: debug, info, warning, error
        - message: string representation of event message
        - scope: optional URL, which points to the loggable instance

        Request example:

        .. code-block:: javascript

            POST /api/events/
            Accept: application/json
            Content-Type: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "level": "info",
                "message": "message#1",
                "scope": "http://example.com/api/customers/9cd869201e1b4158a285427fcd790c1c/"
            }
        """
        self.queryset = self.filter_queryset(self.get_queryset())

        page = self.paginate_queryset(self.queryset)
        if page is not None:
            return self.get_paginated_response(page)
        return response.Response(self.queryset)