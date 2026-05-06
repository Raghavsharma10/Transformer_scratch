def send_command_return(self, obj, command, *arguments):
        """ Send command with single line output.

        :param obj: requested object.
        :param command: command to send.
        :param arguments: list of command arguments.
        :return: command output.
        """
        return self._perform_command('{}/{}'.format(self.session_url, obj.ref), command, OperReturnType.line_output,
                                     *arguments).json()