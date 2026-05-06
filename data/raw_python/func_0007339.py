def create_password(self, data):
        """Create a password."""
        # http://teampasswordmanager.com/docs/api-passwords/#create_password
        log.info('Create new password %s' % data)
        NewID = self.post('passwords.json', data).get('id')
        log.info('Password has been created with ID %s' % NewID)
        return NewID