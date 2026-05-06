def create_default_views(self, create_datastore_views=False):
        # type: (bool) -> None
        """Create default resource views for all resources in dataset

        Args:
            create_datastore_views (bool): Whether to try to create resource views that point to the datastore

        Returns:
            None
        """
        package = deepcopy(self.data)
        if self.resources:
            package['resources'] = self._convert_hdxobjects(self.resources)

        data = {'package': package, 'create_datastore_views': create_datastore_views}
        self._write_to_hdx('create_default_views', data, 'package')