def filter(self, request, queryset, view):
        """ Filter each resource separately using its own filter """
        summary_queryset = queryset
        filtered_querysets = []
        for queryset in summary_queryset.querysets:
            filter_class = self._get_filter(queryset)
            queryset = filter_class(request.query_params, queryset=queryset).qs
            filtered_querysets.append(queryset)

        summary_queryset.querysets = filtered_querysets
        return summary_queryset