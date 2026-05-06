def update_user(self, ID, data):
        """Update a User."""
        # http://teampasswordmanager.com/docs/api-users/#update_user
        log.info('Update user %s with %s' % (ID, data))
        self.put('users/%s.json' % ID, data)