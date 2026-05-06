def stats(self, request, *args, **kwargs):
        """
        To get count of alerts per severities - run **GET** request against */api/alerts/stats/*.
        This endpoint supports all filters that are available for alerts list (*/api/alerts/*).

        Response example:

        .. code-block:: javascript

            {
                "debug": 2,
                "error": 1,
                "info": 1,
                "warning": 1
            }
        """
        queryset = self.filter_queryset(self.get_queryset())
        alerts_severities_count = queryset.values('severity').annotate(count=Count('severity'))

        severity_names = dict(models.Alert.SeverityChoices.CHOICES)
        # For consistency with all other endpoint we need to return severity names in lower case.
        alerts_severities_count = {
            severity_names[asc['severity']].lower(): asc['count'] for asc in alerts_severities_count}
        for severity_name in severity_names.values():
            if severity_name.lower() not in alerts_severities_count:
                alerts_severities_count[severity_name.lower()] = 0

        return response.Response(alerts_severities_count, status=status.HTTP_200_OK)