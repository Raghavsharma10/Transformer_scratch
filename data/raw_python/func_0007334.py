def create_project(self, data):
        """Create a project."""
        # http://teampasswordmanager.com/docs/api-projects/#create_project
        log.info('Create project: %s' % data)
        NewID = self.post('projects.json', data).get('id')
        log.info('Project has been created with ID %s' % NewID)
        return NewID