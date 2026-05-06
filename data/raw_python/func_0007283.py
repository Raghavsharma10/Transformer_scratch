def _save_token_on_disk(self):
        """Helper function that saves the token on disk"""
        token = self._token.copy()

        # Client secret is needed for token refreshing and isn't returned
        # as a pared of OAuth token by default
        token.update(client_secret=self._client_secret)

        with codecs.open(config.TOKEN_FILE_PATH, 'w', 'utf8') as f:
            json.dump(
                token, f,
                ensure_ascii=False,
                sort_keys=True,
                indent=4,
            )