def _run_blocking(self, args, input=None):
        """Internal API: run a blocking command with subprocess.

        This closes any open non-blocking dialog before running the command.

        Parameters
        ----------
        args: Popen constructor arguments
            Command to run.
        input: string
            Value to feed to the stdin of the process.

        Returns
        -------
        (returncode, stdout)
            The exit code (integer) and stdout value (string) from the process.

        """
        # Close any existing dialog.
        if self._process:
            self.close()

        # Make sure we grab stdout as text (not bytes).
        kwargs = {}
        kwargs['stdout'] = subprocess.PIPE
        kwargs['universal_newlines'] = True

        # Use the run() method if available (Python 3.5+).
        if hasattr(subprocess, 'run'):
            result = subprocess.run(args, input=input, **kwargs)
            return result.returncode, result.stdout

        # Have to do our own. If we need to feed stdin, we must open a pipe.
        if input is not None:
            kwargs['stdin'] = subprocess.PIPE

        # Start the process.
        with Popen(args, **kwargs) as proc:
            # Talk to it (no timeout). This will wait until termination.
            stdout, stderr = proc.communicate(input)

            # Find out the return code.
            returncode = proc.poll()

            # Done.
            return returncode, stdout