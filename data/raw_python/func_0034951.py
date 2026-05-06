def _authorization_headers_valid(self, token_type, token):
        """Verify authorization headers for a request.
        Parameters
            token_type (str)
                Type of token to access resources.
            token (str)
                Server Token or OAuth 2.0 Access Token.
        Returns
            (bool)
                True iff token_type and token are valid.
        """
        if token_type not in http.VALID_TOKEN_TYPES:
            return False

        allowed_chars = ascii_letters + digits + '_' + '-' + '=' + '/' + '+'

        # True if token only contains allowed_chars
        return all(characters in allowed_chars for characters in token)