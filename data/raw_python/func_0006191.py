def get_host_for_command(self, command, args):
        """Returns the host this command should be executed against."""
        return self.get_host_for_key(self.get_key(command, args))