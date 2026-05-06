def get_access_token(self, code):
        """Get new access token."""
        try:
            self._token = super().fetch_token(
                MINUT_TOKEN_URL,
                client_id=self._client_id,
                client_secret=self._client_secret,
                code=code,
            )
        # except Exception as e:
        except MissingTokenError as error:
            _LOGGER.debug("Token issues: %s", error)
        return self._token