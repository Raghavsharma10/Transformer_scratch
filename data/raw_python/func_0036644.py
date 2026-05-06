def status(self, value):
        """
            Property for getting or setting the bug status

            >>> bug.status = "REOPENED"
        """
        if self._bug.get('id', None):
            if value in VALID_STATUS:
                self._bug['status'] = value
            else:
                raise BugException("Invalid status type was used")
        else:
            raise BugException("Can not set status unless there is a bug id."
                               " Please call Update() before setting")