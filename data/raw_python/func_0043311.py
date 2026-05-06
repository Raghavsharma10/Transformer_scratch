def revive(self, timeout=None):
        """
        Revive the timeout.

        :param timeout: If not `None`, specifies a new timeout value to use.
        """
        if timeout is not None:
            self.timeout = timeout

        self.revive_event.set()