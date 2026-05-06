def get_push_command(self, remote=None, revision=None):
        """Get the command to push changes from the local repository to a remote repository."""
        if revision:
            raise NotImplementedError(compact("""
                Bazaar repository support doesn't include
                the ability to push specific revisions!
            """))
        command = ['bzr', 'push']
        if remote:
            command.append(remote)
        return command