def revoke(self):
        """Revoke the current Authorization."""
        if self.access_token is None:
            raise InvalidInvocation("no token available to revoke")

        self._authenticator.revoke_token(self.access_token, "access_token")
        self._clear_access_token()