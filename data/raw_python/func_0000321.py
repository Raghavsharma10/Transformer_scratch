def send_command(self, obj, command, *arguments):
        """ Send command and do not parse output (except for communication errors).

        :param obj: requested object.
        :param command: command to send.
        :param arguments: list of command arguments.
        """
        index_command = obj._build_index_command(command, *arguments)
        self.chassis_list[obj.chassis].sendQueryVerify(index_command)