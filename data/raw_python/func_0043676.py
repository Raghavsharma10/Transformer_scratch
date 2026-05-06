def get_push_command(self, remote=None, revision=None):
        """Get the command to push changes from the local repository to a remote repository."""
        # TODO What about tags?
        command = ['git', '-c', 'push.default=matching', 'push']
        if remote or revision:
            command.append(remote or 'origin')
            if revision:
                command.append(revision)
        return command