def is_expired_token(self, client):
        """
        For a given client will test whether or not the token
        has expired.

        This is for testing a client object and does not look up
        from client_id.  You can use _get_client_from_cache() to
        lookup a client from client_id.
        """
        if 'expires' not in client:
            return True

        expires = dateutil.parser.parse(client['expires'])
        if expires < datetime.datetime.now():
            return True

        return False