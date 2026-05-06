def stage_all(self):
        """
        Stages all changed and untracked files
        """
        LOGGER.info('Staging all files')
        self.repo.git.add(A=True)