def is_running(self):
        """Property method that returns a bool specifying if the process is
        currently running. This will return true if the state is active, idle
        or initializing.

        :rtype: bool

        """
        return self._state in [self.STATE_ACTIVE,
                               self.STATE_IDLE,
                               self.STATE_INITIALIZING]