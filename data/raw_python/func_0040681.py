def list_branches(self) -> typing.List[str]:
        """
        :return: branches names
        :rtype: list of str
        """
        branches: typing.List[str] = [head.name for head in self.repo.heads]
        LOGGER.debug('branches: %s', branches)
        return branches