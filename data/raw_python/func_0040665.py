def unstash(self):
        """
        Pops the last stash if EPAB made a stash before
        """
        if not self.stashed:
            LOGGER.error('no stash')
        else:
            LOGGER.info('popping stash')
            self.repo.git.stash('pop')
            self.stashed = False