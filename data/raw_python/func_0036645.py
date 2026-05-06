def update(self):
        """
            Update this object with the latest changes from Bugzilla

            >>> bug.status
            'NEW'
            #Changes happen on Bugzilla
            >>> bug.update()
            >>> bug.status
            'FIXED'
        """
        if 'id' in self._bug:
            result = self._bugsy.request('bug/%s' % self._bug['id'])
            self._bug = dict(**result['bugs'][0])
        else:
            raise BugException("Unable to update bug that isn't in Bugzilla")