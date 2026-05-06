def query_insert(self, direction='max'):
        """
        Query to get latest (default) or earliest update to repository
        """
        if direction == 'min':
            return Layer.objects.aggregate(
                Min('last_updated'))['last_updated__min'].strftime('%Y-%m-%dT%H:%M:%SZ')
        return self._get_repo_filter(Layer.objects).aggregate(
            Max('last_updated'))['last_updated__max'].strftime('%Y-%m-%dT%H:%M:%SZ')