def update_project(self, ID, data):
        """Update a project."""
        # http://teampasswordmanager.com/docs/api-projects/#update_project
        log.info('Update project %s with %s' % (ID, data))
        self.put('projects/%s.json' % ID, data)