def has_command(self, command):
        """Returns True if any of the plugins have the given command."""
        for pbt in self._plugins.values():
            if pbt.command == command:
                return True
        return False