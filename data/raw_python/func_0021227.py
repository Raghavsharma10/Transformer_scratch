def revoke(self, only_access=False):
        """Revoke the current Authorization.

        :param only_access: (Optional) When explicitly set to True, do not
            evict the refresh token if one is set.

        Revoking a refresh token will in-turn revoke all access tokens
        associated with that authorization.

        """
        if only_access or self.refresh_token is None:
            super(Authorizer, self).revoke()
        else:
            self._authenticator.revoke_token(
                self.refresh_token, "refresh_token"
            )
            self._clear_access_token()
            self.refresh_token = None