def list_staged_files(self) -> typing.List[str]:
        """
        :return: staged files
        :rtype: list of str
        """
        staged_files: typing.List[str] = [x.a_path for x in self.repo.index.diff('HEAD')]
        LOGGER.debug('staged files: %s', staged_files)
        return staged_files