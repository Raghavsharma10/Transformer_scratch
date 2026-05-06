def commands(cls):
        """Returns a list of all methods that start with ``cmd_``."""
        cmds = [cmd[4:] for cmd in dir(cls) if cmd.startswith('cmd_')]
        return cmds