def revoke_token(self, token, token_type=None):
        """Ask Reddit to revoke the provided token.

        :param token: The access or refresh token to revoke.
        :param token_type: (Optional) When provided, hint to Reddit what the
            token type is for a possible efficiency gain. The value can be
            either ``access_token`` or ``refresh_token``.

        """
        data = {"token": token}
        if token_type is not None:
            data["token_type_hint"] = token_type
        url = self._requestor.reddit_url + const.REVOKE_TOKEN_PATH
        self._post(url, success_status=codes["no_content"], **data)