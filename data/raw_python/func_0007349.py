def change_user_password(self, ID, data):
        """Change password of a User."""
        # http://teampasswordmanager.com/docs/api-users/#change_password
        log.info('Change user %s password' % ID)
        self.put('users/%s/change_password.json' % ID, data)