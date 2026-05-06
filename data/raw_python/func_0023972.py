def _exchange_refresh_tokens(self):
        'Exchanges a refresh token for an access token'
        if self.token_cache is not None and 'refresh' in self.token_cache:
            # Attempt to use the refresh token to get a new access token.
            refresh_form = {
                'grant_type': 'refresh_token',
                'refresh_token': self.token_cache['refresh'],
                'client_id': self.client_id,
                'client_secret': self.client_secret,
            }
            try:
                tokens = self._request_tokens_from_token_endpoint(refresh_form)
                tokens['refresh'] = self.token_cache['refresh']
                return tokens
            except OAuth2Exception:
                logging.exception(
                    'Encountered an exception during refresh token flow.')
        return None