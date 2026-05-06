def revoke(self, token_id, *args, **kwargs):
        """
        Revokes a personal access token.
        """

        return self.delete(token_id, *args, **kwargs)