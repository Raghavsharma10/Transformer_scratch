def update_mypassword(self, ID, data):
        """Update my password."""
        # http://teampasswordmanager.com/docs/api-my-passwords/#update_password
        log.info('Update MyPassword %s with %s' % (ID, data))
        self.put('my_passwords/%s.json' % ID, data)