def users(self, request, uuid=None):
        """ A list of users connected to the project """
        project = self.get_object()
        queryset = project.get_users()
        # we need to handle filtration manually because we want to filter only project users, not projects.
        filter_backend = filters.UserConcatenatedNameOrderingBackend()
        queryset = filter_backend.filter_queryset(request, queryset, self)
        queryset = self.paginate_queryset(queryset)
        serializer = self.get_serializer(queryset, many=True)
        return self.get_paginated_response(serializer.data)