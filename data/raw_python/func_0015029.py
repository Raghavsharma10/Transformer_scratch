def cmd_slow_requests(self):
        """List all requests that took a certain amount of time to be
        processed.

        .. warning::
           By now hardcoded to 1 second (1000 milliseconds), improve the
           command line interface to allow to send parameters to each command
           or globally.
        """
        slow_requests = [
            line.time_wait_response
            for line in self._valid_lines
            if line.time_wait_response > 1000
        ]
        return slow_requests