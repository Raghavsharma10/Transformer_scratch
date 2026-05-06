def history(self, request, uuid=None):
        """
        Historical data endpoints could be available for any objects (currently
        implemented for quotas and events count). The data is available at *<object_endpoint>/history/*,
        for example: */api/quotas/<uuid>/history/*.

        There are two ways to define datetime points for historical data.

        1. Send *?point=<timestamp>* parameter that can list. Response will contain historical data for each given point
            in the same order.
        2. Send *?start=<timestamp>*, *?end=<timestamp>*, *?points_count=<integer>* parameters.
           Result will contain <points_count> points from <start> to <end>.

        Response format:

        .. code-block:: javascript

            [
                {
                    "point": <timestamp>,
                    "object": {<object_representation>}
                },
                {
                    "point": <timestamp>
                    "object": {<object_representation>}
                },
            ...
            ]

        NB! There will not be any "object" for corresponding point in response if there
        is no data about object for a given timestamp.
        """
        mapped = {
            'start': request.query_params.get('start'),
            'end': request.query_params.get('end'),
            'points_count': request.query_params.get('points_count'),
            'point_list': request.query_params.getlist('point'),
        }
        history_serializer = HistorySerializer(data={k: v for k, v in mapped.items() if v})
        history_serializer.is_valid(raise_exception=True)

        quota = self.get_object()
        serializer = self.get_serializer(quota)
        serialized_versions = []
        for point_date in history_serializer.get_filter_data():
            serialized = {'point': datetime_to_timestamp(point_date)}
            version = Version.objects.get_for_object(quota).filter(revision__date_created__lte=point_date)
            if version.exists():
                # make copy of serialized data and update field that are stored in version
                version_object = version.first()._object_version.object
                serialized['object'] = serializer.data.copy()
                serialized['object'].update({
                    f: getattr(version_object, f) for f in quota.get_version_fields()
                })
            serialized_versions.append(serialized)
        return response.Response(serialized_versions, status=status.HTTP_200_OK)