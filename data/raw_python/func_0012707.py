def _get_process(self):
        """
        Create the process by running the specified command.
        """
        command = self._get_command()
        return subprocess.Popen(command, bufsize=-1, close_fds=True,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE)