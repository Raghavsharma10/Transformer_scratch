def derive_queryset(self, **kwargs):
        """
        Derives our queryset.
        """
        # get our parent queryset
        queryset = super(SmartListView, self).get_queryset(**kwargs)

        # apply any filtering
        search_fields = self.derive_search_fields()
        search_query = self.request.GET.get('search')
        if search_fields and search_query:
            term_queries = []
            for term in search_query.split(' '):
                field_queries = []
                for field in search_fields:
                    field_queries.append(Q(**{field: term}))
                term_queries.append(reduce(operator.or_, field_queries))

            queryset = queryset.filter(reduce(operator.and_, term_queries))

        # add any select related
        related = self.derive_select_related()
        if related:
            queryset = queryset.select_related(*related)

        # return our queryset
        return queryset