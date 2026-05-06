def _run_nonblocking(self, args, input=None):
        """Internal API: run a non-blocking command with subprocess.

        This closes any open non-blocking dialog before running the command.

        Parameters
        ----------
        args: Popen constructor arguments
            Command to run.
        input: string
            Value to feed to the stdin of the process.

        """
        # Close any existing dialog.
        if self._process:
            self.close()

        # Start the new one.
        self._process = subprocess.Popen(args, stdout=subprocess.PIPE)