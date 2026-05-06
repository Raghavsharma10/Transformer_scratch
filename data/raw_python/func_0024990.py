def _get_policy_set_uri(self, guid=None):
        """
        Returns the full path that uniquely identifies
        the subject endpoint.
        """
        uri = self.uri + '/v1/policy-set'
        if guid:
            uri += '/' + urllib.quote_plus(guid)
        return uri