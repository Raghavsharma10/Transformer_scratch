def fetchall(self):
        """Fetch all refs from the upstream repo."""
        try:
            self.repo.remotes.origin.fetch()
        except git.exc.GitCommandError as err:
            raise GitError(err)