def die(self):
        """Stops the server if it is running."""
        if self.process:
            _log(self.logging,
                 'Stopping {0} server with PID: {1} running at {2}.'
                     .format(self.__class__.__name__, self.process.pid,
                             self.check_url))

            self._kill()