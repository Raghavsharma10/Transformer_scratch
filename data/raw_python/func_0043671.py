def get_commit_command(self, message, author=None):
        """Get the command to commit changes to tracked files in the working tree."""
        command = ['git']
        if author:
            command.extend(('-c', 'user.name=%s' % author.name))
            command.extend(('-c', 'user.email=%s' % author.email))
        command.append('commit')
        command.append('--all')
        command.append('--message')
        command.append(message)
        return command