def get_resource(self, resource_id):
        """
        Returns a specific resource by resource id.
        """
        # resource_id could be a path such as '/asset/123' so quote
        uri = self._get_resource_uri(guid=resource_id)
        return self.service._get(uri)