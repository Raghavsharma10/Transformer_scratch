def act(self, cmd_name, params=None):
        """ Run the specified command with its parameters."""

        command = getattr(self, cmd_name)
        if params:
            command(params)
        else:
            command()