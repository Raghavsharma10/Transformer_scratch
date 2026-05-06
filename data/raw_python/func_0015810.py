def set_password(self, service, username, password):
        """Set password for the username of the service
        """
        segments = range(0, len(password), self._max_password_size)
        password_parts = [
            password[i:i + self._max_password_size] for i in segments]
        for i, password_part in enumerate(password_parts):
            curr_username = username
            if i > 0:
                curr_username += '{{part_%d}}' % i
            self._keyring.set_password(service, curr_username, password_part)