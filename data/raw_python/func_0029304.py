def _get_credentials(self, request):
        """ Extract username and api key token from 'Authorization' header """
        authorization = request.headers.get('Authorization')
        if not authorization:
            return None
        try:
            authmeth, authbytes = authorization.split(' ', 1)
        except ValueError:  # not enough values to unpack
            return None
        if authmeth.lower() != 'apikey':
            return None

        if six.PY2 or isinstance(authbytes, bytes):
            try:
                auth = authbytes.decode('utf-8')
            except UnicodeDecodeError:
                auth = authbytes.decode('latin-1')
        else:
            auth = authbytes

        try:
            username, api_key = auth.split(':', 1)
        except ValueError:  # not enough values to unpack
            return None
        return username, api_key