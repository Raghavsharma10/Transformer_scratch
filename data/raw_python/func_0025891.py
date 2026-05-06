def close(self, notes=None):
        '''Save all changes and close this issue'''
        self.set_status(self._redmine.ISSUE_STATUS_ID_CLOSED, notes=notes)