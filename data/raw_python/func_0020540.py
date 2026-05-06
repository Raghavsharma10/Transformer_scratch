def retries_disabled(self):
        """
        Context manager to disable retries on requests
        :returns: OSBS object
        """
        self.os.retries_enabled = False
        yield
        self.os.retries_enabled = True