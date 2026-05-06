def update_security_of_password(self, ID, data):
        """Update security of a password."""
        # http://teampasswordmanager.com/docs/api-passwords/#update_security_password
        log.info('Update security of password %s with %s' % (ID, data))
        self.put('passwords/%s/security.json' % ID, data)