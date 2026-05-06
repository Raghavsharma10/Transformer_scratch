def _add_bearer_token(self, *args, **kwargs):
        """Add a bearer token to the request uri, body or authorization header.

        This is overwritten to change the headers slightly.
        """
        s = super(TwitchOAuthClient, self)
        uri, headers, body = s._add_bearer_token(*args, **kwargs)
        authheader = headers.get('Authorization')
        if authheader:
            headers['Authorization'] = authheader.replace('Bearer', 'OAuth')
        return uri, headers, body