def update_password(self, ID, data):
        """Update a password."""
        # http://teampasswordmanager.com/docs/api-passwords/#update_password
        log.info('Update Password %s with %s' % (ID, data))
        self.put('passwords/%s.json' % ID, data)