def no_auth(self):
        """Unset authentication temporarily as a context manager."""
        old_basic_auth, self.auth = self.auth, None
        old_token_auth = self.headers.pop('Authorization', None)

        yield

        self.auth = old_basic_auth
        if old_token_auth:
            self.headers['Authorization'] = old_token_auth