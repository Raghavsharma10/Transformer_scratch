def authenticate(self, request, username=None, password=None):
        """
        Check credentials against RADIUS server and return a User object or
        None.
        """
        if isinstance(username, basestring):
            username = username.encode('utf-8')

        if isinstance(password, basestring):
            password = password.encode('utf-8')

        server = self._get_server_from_settings()
        result = self._radius_auth(server, username, password)

        if result:
            return self.get_django_user(username, password)

        return None