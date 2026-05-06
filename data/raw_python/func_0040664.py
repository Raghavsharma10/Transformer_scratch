def stash(self, stash_name: str):
        """
        Stashes the current working tree changes

        :param stash_name: name of the stash
        :type stash_name: str
        """
        if self.stashed:
            LOGGER.error('already stashed')
            sys.exit(-1)
        else:
            if not self.index_is_empty():
                LOGGER.error('cannot stash; index is not empty')
                sys.exit(-1)
            if self.untracked_files():
                LOGGER.error('cannot stash; there are untracked files')
                sys.exit(-1)
            if self.changed_files():
                LOGGER.info('stashing changes')
                self.repo.git.stash('push', '-u', '-k', '-m', f'"{stash_name}"')
                self.stashed = True
            else:
                LOGGER.info('no changes to stash')