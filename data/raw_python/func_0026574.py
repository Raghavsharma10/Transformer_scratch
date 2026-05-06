def users_list(self, *args):
        """Display a list of connected users"""
        if len(self._users) == 0:
            self.log('No users connected')
        else:
            self.log(self._users, pretty=True)