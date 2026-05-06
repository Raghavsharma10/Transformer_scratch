def update_security_of_project(self, ID, data):
        """Update security of project."""
        # http://teampasswordmanager.com/docs/api-projects/#update_project_security
        log.info('Update project %s security %s' % (ID, data))
        self.put('projects/%s/security.json' % ID, data)