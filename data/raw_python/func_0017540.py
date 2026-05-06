def get_resource_metadata(self, resource=None):
        """
        Get resource metadata
        :param resource: The name of the resource to get metadata for
        :return: list
        """
        result = self._make_metadata_request(meta_id=0, metadata_type='METADATA-RESOURCE')
        if resource:
            result = next((item for item in result if item['ResourceID'] == resource), None)
        return result