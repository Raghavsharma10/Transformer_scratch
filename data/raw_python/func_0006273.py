def run_command(self, command):
        """ Run a command.

        :arg str command: The command to run.

        :Return: The result as a string.

        :Raises: `ValueError` if `command` is not a string.

        """
        if not isinstance(command, basestring):
            raise ValueError("command must be a string")
        return self._ssh_client.run_gerrit_command(command)