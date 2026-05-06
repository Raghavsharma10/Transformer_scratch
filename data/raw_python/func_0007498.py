def fetch_by_code(self, code):
        """
        Returns data belonging to an authorization code from memcache or
        ``None`` if no data was found.

        See :class:`oauth2.store.AuthCodeStore`.

        """
        code_data = self.mc.get(self._generate_cache_key(code))

        if code_data is None:
            raise AuthCodeNotFound

        return AuthorizationCode(**code_data)