def _get_resource_uri(self, guid=None):
        """
        Returns the full path that uniquely identifies
        the resource endpoint.
        """
        uri = self.uri + '/v1/resource'
        if guid:
            uri += '/' + urllib.quote_plus(guid)
        return uri