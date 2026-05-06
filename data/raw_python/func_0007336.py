def change_parent_of_project(self, ID, NewParrentID):
        """Change parent of project."""
        # http://teampasswordmanager.com/docs/api-projects/#change_parent
        log.info('Change parrent for project %s to %s' % (ID, NewParrentID))
        data = {'parent_id': NewParrentID}
        self.put('projects/%s/change_parent.json' % ID, data)