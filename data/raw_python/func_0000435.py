def send_command_return_multilines(self, obj, command, *arguments):
        """ Send command with no output.

        :param obj: requested object.
        :param command: command to send.
        :param arguments: list of command arguments.
        :return: list of command output lines.
        :rtype: list(str)
        """
        return self._perform_command('{}/{}'.format(self.session_url, obj.ref), command,
                                     OperReturnType.multiline_output, *arguments).json()