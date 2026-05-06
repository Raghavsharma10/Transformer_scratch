def callback(self, username, request):
        """ Having :username: return user's identifiers or None. """
        credentials = self._get_credentials(request)
        if credentials:
            username, api_key = credentials
            if self.check:
                return self.check(username, api_key, request)