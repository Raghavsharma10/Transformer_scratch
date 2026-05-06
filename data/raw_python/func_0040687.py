def is_dirty(self, untracked=False) -> bool:
        """
        Checks if the current repository contains uncommitted or untracked changes

        Returns: true if the repository is clean
        """
        result = False
        if not self.index_is_empty():
            LOGGER.error('index is not empty')
            result = True
        changed_files = self.changed_files()
        if bool(changed_files):

            LOGGER.error(f'Repo has %s modified files: %s', len(changed_files), changed_files)
            result = True
        if untracked:
            result = result or bool(self.untracked_files())
        return result