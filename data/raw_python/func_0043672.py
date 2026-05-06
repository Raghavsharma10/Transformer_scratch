def get_create_command(self):
        """Get the command to create the local repository."""
        command = ['git', 'clone' if self.remote else 'init']
        if self.bare:
            command.append('--bare')
        if self.remote:
            command.append(self.remote)
        command.append(self.local)
        return command