def delete_policy_set(self, policy_set_id):
        """
        Delete a specific policy set by id.  Method is idempotent.
        """
        uri = self._get_policy_set_uri(guid=policy_set_id)
        return self.service._delete(uri)