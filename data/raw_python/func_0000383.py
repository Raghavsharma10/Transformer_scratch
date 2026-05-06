def send_command(self, command, *arguments):
        """ Send command with no output.

        :param command: command to send.
        :param arguments: list of command arguments.
        """
        self.api.send_command(self, command, *arguments)