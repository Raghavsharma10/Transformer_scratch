def push(self, set_upstream: bool = True):
        """
        Pushes all refs (branches and tags) to origin
        """
        LOGGER.info('pushing repo to origin')

        try:
            self.repo.git.push()
        except GitCommandError as error:
            if 'has no upstream branch' in error.stderr and set_upstream:
                self.repo.git.push(f'--set-upstream origin {self.get_current_branch()}')
            else:
                raise
        self.push_tags()