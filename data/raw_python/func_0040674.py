def stage_modified(self):
        """
        Stages modified files only (no untracked)
        """
        LOGGER.info('Staging modified files')
        self.repo.git.add(u=True)