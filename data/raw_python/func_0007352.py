def update_group(self, ID, data):
        """Update a Group."""
        # http://teampasswordmanager.com/docs/api-groups/#update_group
        log.info('Update group %s with %s' % (ID, data))
        self.put('groups/%s.json' % ID, data)