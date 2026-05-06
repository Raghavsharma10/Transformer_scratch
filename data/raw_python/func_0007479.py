def fetch_by_refresh_token(self, refresh_token):
        """
        Find an access token by its refresh token.

        :param refresh_token: The refresh token that was assigned to an
                              ``AccessToken``.
        :return: The :class:`oauth2.datatype.AccessToken`.
        :raises: :class:`oauth2.error.AccessTokenNotFound`
        """
        if refresh_token not in self.refresh_tokens:
            raise AccessTokenNotFound

        return self.refresh_tokens[refresh_token]