def basicauthfail(self, realm = b'all'):
        """
        Return 401 for authentication failure. This will end the handler.
        """
        if not isinstance(realm, bytes):
            realm = realm.encode('ascii')
        self.start_response(401, [(b'WWW-Authenticate', b'Basic realm="' + realm + b'"')])
        self.exit(b'<h1>' + _createstatus(401) + b'</h1>')