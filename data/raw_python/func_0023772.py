def ensure_started(self):
        """
        Start a server and waits (blocking wait) until it is fully started.
        """
        # server is either starting or stopping (or error)
        if self.state in ['maintenance', 'error']:
            self._wait_for_state_change(['stopped', 'started'])

        if self.state == 'stopped':
            self.start()
            self._wait_for_state_change(['started'])

        if self.state == 'started':
            return True
        else:
            # something went wrong, fail explicitly
            raise Exception('unknown server state: ' + self.state)