def generate_token(self, username):
        """
        assumes user exists in htpasswd file.

        Return the token for the given user by signing a token of
        the username and a hash of the htpasswd string.
        """
        serializer = self.get_signature()
        return serializer.dumps(
            {
                'username': username,
                'hashhash': self.get_hashhash(username)
            }
        ).decode('UTF-8')