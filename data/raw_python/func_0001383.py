def remove(self, username=None):
        """Remove User instance based on supplied user name."""
        self._user_list = [user for user in self._user_list if user.name != username]