def check_token_auth(self, token):
        """
        Check to see who this is and if their token gets
        them into the system.
        """
        serializer = self.get_signature()

        try:
            data = serializer.loads(token)
        except BadSignature:
            log.warning('Received bad token signature')
            return False, None
        if data['username'] not in self.users.users():
            log.warning(
                'Token auth signed message, but invalid user %s',
                data['username']
            )
            return False, None
        if data['hashhash'] != self.get_hashhash(data['username']):
            log.warning(
                'Token and password do not match, %s '
                'needs to regenerate token',
                data['username']
            )
            return False, None
        return True, data['username']