def count(self, request, *args, **kwargs):
        """
        To get a count of events - run **GET** against */api/events/count/* as authenticated user.
        Endpoint support same filters as events list.

        Response example:

        .. code-block:: javascript

            {"count": 12321}
        """

        self.queryset = self.filter_queryset(self.get_queryset())
        return response.Response({'count': self.queryset.count()}, status=status.HTTP_200_OK)