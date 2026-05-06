def _put_resource(self, resource_id, body):
        """
        Update a resource for the given resource id.  The body is not
        a list but a dictionary of a single resource.
        """
        assert isinstance(body, (dict)), "PUT requires body to be a dict."
        # resource_id could be a path such as '/asset/123' so quote
        uri = self._get_resource_uri(guid=resource_id)
        return self.service._put(uri, body)