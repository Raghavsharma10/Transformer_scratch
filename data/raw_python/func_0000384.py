def send_command_return(self, command, *arguments):
        """ Send command and wait for single line output. """
        return self.api.send_command_return(self, command, *arguments)