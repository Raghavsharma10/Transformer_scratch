def create_mypassword(self, data):
        """Create my password."""
        # http://teampasswordmanager.com/docs/api-my-passwords/#create_password
        log.info('Create MyPassword with %s' % data)
        NewID = self.post('my_passwords.json', data).get('id')
        log.info('MyPassword has been created with %s' % NewID)
        return NewID