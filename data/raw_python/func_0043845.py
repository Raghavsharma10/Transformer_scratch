def get_checkout_command(self, revision, clean=False):
        """Get the command to update the working tree of the local repository."""
        command = ['hg', 'update']
        if clean:
            command.append('--clean')
        command.append('--rev=%s' % revision)
        return command