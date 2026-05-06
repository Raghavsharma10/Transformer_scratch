def list_versions(self):
        """
        Filterable list of versions of a layer, always ordered newest to oldest.

        If the version’s source supports revisions, you can get a specific revision using
        ``.filter(data__source__revision=value)``. Specific values depend on the source type.
        Use ``data__source_revision__lt`` or ``data__source_revision__gte`` to filter
        using ``<`` or ``>=`` operators respectively.
        """
        target_url = self._client.get_url('VERSION', 'GET', 'multi', {'layer_id': self.id})
        return base.Query(self._manager, target_url, valid_filter_attributes=('data',), valid_sort_attributes=())