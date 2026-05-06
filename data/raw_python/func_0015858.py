def get_password(self, service, username):
        """Get password of the username for the service
        """
        result = self._get_entry(self._keyring, service, username)
        if result:
            result = self._decrypt(result)
        return result