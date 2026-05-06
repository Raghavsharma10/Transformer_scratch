def add_command(self, cmd, name=None):
        """Registers another :class:`Command` with this group.  If the name
        is not provided, the name of the command is used.
        """
        name = name or cmd.name
        if name is None:
            raise TypeError('Command has no name.')
        self.commands[name] = cmd