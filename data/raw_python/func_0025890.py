def resolve(self, notes=None):
        '''Save all changes and resolve this issue'''
        self.set_status(self._redmine.ISSUE_STATUS_ID_RESOLVED, notes=notes)