def list(self, request, *args, **kwargs):
        """Filter Content with a custom search.
        {
            "query": SEARCH_QUERY
            "preview": true
        }
        "preview" is optional and, when true, will include
        items that would normally be removed due to "excluded_ids".
        """

        queryset = self.get_filtered_queryset(get_request_data(request))
        # Switch between paginated or standard style responses
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)