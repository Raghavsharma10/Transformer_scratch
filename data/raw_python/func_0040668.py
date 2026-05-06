def untracked_files(self) -> typing.List[str]:
        """
        :return: of untracked files
        :rtype: list
        """
        untracked_files = list(self.repo.untracked_files)
        LOGGER.debug('untracked files: %s', untracked_files)
        return untracked_files