def delete_refresh_token(self, refresh_token):
        """
        Deletes a refresh token after use
        :param refresh_token: The refresh token to delete.
        """
        access_token = self.fetch_by_refresh_token(refresh_token)
        self.mc.delete(self._generate_cache_key(access_token.token))
        self.mc.delete(self._generate_cache_key(refresh_token))