def count_history(self, request, *args, **kwargs):
        """
        To get a historical data of events amount - run **GET** against */api/events/count/history/*.
        Endpoint support same filters as events list.
        More about historical data - read at section *Historical data*.

        Response example:

        .. code-block:: javascript

            [
                {
                    "point": 141111111111,
                    "object": {
                        "count": 558
                    }
                }
            ]
        """
        queryset = self.filter_queryset(self.get_queryset())

        mapped = {
            'start': request.query_params.get('start'),
            'end': request.query_params.get('end'),
            'points_count': request.query_params.get('points_count'),
            'point_list': request.query_params.getlist('point'),
        }
        serializer = core_serializers.HistorySerializer(data={k: v for k, v in mapped.items() if v})
        serializer.is_valid(raise_exception=True)

        timestamp_ranges = [{'end': point_date} for point_date in serializer.get_filter_data()]
        aggregated_count = queryset.aggregated_count(timestamp_ranges)

        return response.Response(
            [{'point': int(ac['end']), 'object': {'count': ac['count']}} for ac in aggregated_count],
            status=status.HTTP_200_OK)