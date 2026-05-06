def create_user(self, data):
        """Create a User."""
        # http://teampasswordmanager.com/docs/api-users/#create_user
        log.info('Create user with %s' % data)
        NewID = self.post('users.json', data).get('id')
        log.info('User has been created with ID %s' % NewID)
        return NewID