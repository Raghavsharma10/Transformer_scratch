def _checkPermissions(self, user, event):
        """Checks if the user has in any role that allows to fire the event."""

        for role in user.account.roles:
            if role in event.roles:
                self.log('Access granted', lvl=verbose)
                return True

        self.log('Access denied', lvl=verbose)
        return False