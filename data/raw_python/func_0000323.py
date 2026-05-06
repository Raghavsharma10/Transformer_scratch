def send_command_return_multilines(self, obj, command, *arguments):
        """ Send command and wait for multiple lines output. """
        index_command = obj._build_index_command(command, *arguments)
        return self.chassis_list[obj.chassis].sendQuery(index_command, True)