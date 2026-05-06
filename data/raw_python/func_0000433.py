def send_command(self, obj, command, *arguments):
        """ Send command with no output.

        :param obj: requested object.
        :param command: command to send.
        :param arguments: list of command arguments.
        """
        self._perform_command('{}/{}'.format(self.session_url, obj.ref), command, OperReturnType.no_output, *arguments)