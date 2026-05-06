def delete_refresh_token(self, refresh_token):
        """
        Deletes a refresh token after use
        :param refresh_token: The refresh token to delete.
        """
        access_token = self.fetch_by_refresh_token(refresh_token)

        self.delete(access_token.token)