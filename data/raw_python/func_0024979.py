def delete_resource(self, resource_id):
        """
        Remove a specific resource by its identifier.
        """
        # resource_id could be a path such as '/asset/123' so quote
        uri = self._get_resource_uri(guid=resource_id)
        return self.service._delete(uri)