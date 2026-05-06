def changed_files(self) -> typing.List[str]:
        """
        :return: changed files
        :rtype: list of str
        """
        changed_files: typing.List[str] = [x.a_path for x in self.repo.index.diff(None)]
        LOGGER.debug('changed files: %s', changed_files)
        return changed_files