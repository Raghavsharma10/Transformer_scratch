def get_lookup_values(self, resource, lookup_name):
        """
        Get possible lookup values for a given field
        :param resource: The name of the resource
        :param lookup_name: The name of the the field to get lookup values for
        :return: list
        """
        return self._make_metadata_request(meta_id=resource + ':' + lookup_name, metadata_type='METADATA-LOOKUP_TYPE')