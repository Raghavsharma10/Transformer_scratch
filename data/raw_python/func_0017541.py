def get_table_metadata(self, resource, resource_class):
        """
        Get metadata for a given resource: class
        :param resource: The name of the resource
        :param resource_class: The name of the class to get metadata from
        :return: list
        """
        return self._make_metadata_request(meta_id=resource + ':' + resource_class, metadata_type='METADATA-TABLE')