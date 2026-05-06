def send_command_return_multilines(self, command, *arguments):
        """ Send command and wait for multiple lines output. """
        return self.api.send_command_return_multilines(self, command, *arguments)