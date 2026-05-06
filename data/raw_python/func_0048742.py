def get_queryset_filters(self, query):
        """
        Return the filtered queryset
        """
        conditions = Q()
        for field_name in self.fields:
            conditions |= Q(**{
                self._construct_qs_filter(field_name): query
            })
        return conditions