def create_group(self, data):
        """Create a Group."""
        # http://teampasswordmanager.com/docs/api-groups/#create_group
        log.info('Create group with %s' % data)
        NewID = self.post('groups.json', data).get('id')
        log.info('Group has been created with ID %s' % NewID)
        return NewID