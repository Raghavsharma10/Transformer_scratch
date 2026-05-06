def list(self, request, *args, **kwargs):
        """Modified list view to driving listing from ES"""
        search_kwargs = {"published": False}

        for field_name in ("before", "after", "status", "published"):

            if field_name in get_query_params(self.request):
                search_kwargs[field_name] = get_query_params(self.request).get(field_name)

        for field_name in ("tags", "types", "feature_types"):

            if field_name in get_query_params(self.request):
                search_kwargs[field_name] = get_query_params(self.request).getlist(field_name)

        if "search" in get_query_params(self.request):
            search_kwargs["query"] = get_query_params(self.request).get("search")

        queryset = self.model.search_objects.search(**search_kwargs)

        if "authors" in get_query_params(self.request):
            authors = get_query_params(self.request).getlist("authors")
            queryset = queryset.filter(Authors(authors))

        if "exclude" in get_query_params(self.request):
            exclude = get_query_params(self.request).get("exclude")
            queryset = queryset.filter(
                es_filter.Not(es_filter.Type(**{'value': exclude}))
            )

        # always filter out Super Features from listing page
        queryset = queryset.filter(
            es_filter.Not(filter=es_filter.Type(
                value=get_superfeature_model().search_objects.mapping.doc_type))
        )

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)