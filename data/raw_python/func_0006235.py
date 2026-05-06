def act(self, cmd_name, param):
        """
            Run the command with the parameters.

        :param cmd_name: The name of command to run
        :param param: Params for the command
        """
        command = getattr(self, cmd_name)
        if param:
            command(param)
        else:
            command()