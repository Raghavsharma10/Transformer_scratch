def execute(self, *args):
        """Executes single command and returns result."""
        command, kwargs = self.parse(*args)
        return self._commands.execute(command, **kwargs)