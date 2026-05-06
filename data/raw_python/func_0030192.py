def auth_as(self, user):
        """auth as a user temporarily"""
        old_user = self._user
        self.auth(user)
        try:
            yield
        finally:
            self.auth(old_user)