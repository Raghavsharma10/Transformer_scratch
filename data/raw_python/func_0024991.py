def _put_policy_set(self, policy_set_id, body):
        """
        Will create or update a policy set for the given path.
        """
        assert isinstance(body, (dict)), "PUT requires body to be a dict."
        uri = self._get_policy_set_uri(guid=policy_set_id)
        return self.service._put(uri, body)