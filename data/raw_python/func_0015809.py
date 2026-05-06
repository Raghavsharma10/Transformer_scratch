def get_password(self, service, username):
        """Get password of the username for the service
        """
        init_part = self._keyring.get_password(service, username)
        if init_part:
            parts = [init_part]
            i = 1
            while True:
                next_part = self._keyring.get_password(
                    service,
                    '%s{{part_%d}}' % (username, i))
                if next_part:
                    parts.append(next_part)
                    i += 1
                else:
                    break
            return ''.join(parts)
        return None