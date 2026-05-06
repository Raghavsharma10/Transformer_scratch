def get_commit_command(self, message, author=None):
        """Get the command to commit changes to tracked files in the working tree."""
        command = ['bzr', 'commit']
        if author:
            command.extend(('--author', author.combined))
        command.append('--message')
        command.append(message)
        return command