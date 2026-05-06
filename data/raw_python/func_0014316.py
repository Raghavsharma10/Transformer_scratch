def delete(self, token_id, *args, **kwargs):
        """
        Revokes a personal access token.
        """

        return self.client._put(
            "{0}/revoked".format(
                self._url(token_id)
            ),
            None,
            *args,
            **kwargs
        )