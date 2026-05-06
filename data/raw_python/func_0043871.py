def get_pull_command(self, remote=None, revision=None):
        """Get the command to pull changes from a remote repository into the local repository."""
        if revision:
            raise NotImplementedError(compact("""
                Bazaar repository support doesn't include
                the ability to pull specific revisions!
            """))
        command = ['bzr', 'pull']
        if remote:
            command.append(remote)
        return command