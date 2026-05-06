def authenticate(self, request, username=None, password=None, realm=None):
        """
        Check credentials against the RADIUS server identified by `realm` and
        return a User object or None. If no argument is supplied, Django will
        skip this backend and try the next one (as a TypeError will be raised
        and caught).
        """
        if isinstance(username, basestring):
            username = username.encode('utf-8')

        if isinstance(password, basestring):
            password = password.encode('utf-8')

        server = self.get_server(realm)

        if not server:
            return None

        result = self._radius_auth(server, username, password)

        if result:
            full_username = self.construct_full_username(username, realm)
            return self.get_django_user(full_username, password)

        return None