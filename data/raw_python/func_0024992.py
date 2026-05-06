def _get_policy_set(self, policy_set_id):
        """
        Get a specific policy set by id.
        """
        uri = self._get_policy_set_uri(guid=policy_set_id)
        return self.service._get(uri)