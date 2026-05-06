def authenticate(self, client_id, client_secret, use_cache=True):
        """
        Authenticate the given client against UAA.  The resulting token
        will be cached for reuse.
        """
        # We will reuse a token for as long as we have one cached
        # and it hasn't expired.
        if use_cache:
            client = self._get_client_from_cache(client_id)
            if (client) and (not self.is_expired_token(client)):
                self.authenticated = True
                self.client = client
                return

        # Let's authenticate the client
        client = {
            'id': client_id,
            'secret': client_secret
        }

        res = self._authenticate_client(client_id, client_secret)
        client.update(res)

        expires = datetime.datetime.now() + \
                  datetime.timedelta(seconds=res['expires_in'])
        client['expires'] = expires.isoformat()

        # Cache it for repeated use until expired
        self._write_to_uaa_cache(client)

        self.client = client
        self.authenticated = True