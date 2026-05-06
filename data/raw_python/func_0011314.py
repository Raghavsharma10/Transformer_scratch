def add_command(self, handler, name=None):
        """Add a subcommand `name` which invokes `handler`.
        """
        if name is None:
            name = docstring_to_subcommand(handler.__doc__)

        # TODO: Prevent overwriting 'help'?
        self._commands[name] = handler