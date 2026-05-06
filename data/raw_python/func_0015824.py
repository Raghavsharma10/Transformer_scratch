def get_password(self, service, username):
        """Get password of the username for the service
        """
        items = self._find_passwords(service, username)
        if not items:
            return None

        secret = items[0].secret
        return (
            secret
            if isinstance(secret, six.text_type) else
            secret.decode('utf-8')
        )