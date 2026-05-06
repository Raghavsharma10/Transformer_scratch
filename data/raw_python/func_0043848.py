def get_pull_command(self, remote=None, revision=None):
        """Get the command to pull changes from a remote repository into the local repository."""
        command = ['hg', 'pull']
        if remote:
            command.append(remote)
        if revision:
            command.append('--rev=%s' % revision)
        return command