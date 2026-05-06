def get_queryset(self, request):
        """Shows one entry per distinct metric name"""
        queryset = super(MetricGroupAdmin, self).get_queryset(request)
        # poor-man's DISTINCT ON for Sqlite3
        qs_values = queryset.values('id', 'name')
        # 2.7+ only :(
        # = {metric['name']: metric['id'] for metric in qs_values}
        distinct_names = {}
        for metric in qs_values:
            distinct_names[metric['name']] = metric['id']
        queryset = self.model.objects.filter(id__in=distinct_names.values())
        return queryset