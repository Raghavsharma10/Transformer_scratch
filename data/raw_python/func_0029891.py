def check_basic_auth(self, username, password):
        """
        This function is called to check if a username /
        password combination is valid via the htpasswd file.
        """
        valid = self.users.check_password(
            username, password
        )
        if not valid:
            log.warning('Invalid login from %s', username)
            valid = False
        return (
            valid,
            username
        )