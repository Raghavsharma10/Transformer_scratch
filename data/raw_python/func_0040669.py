def status(self) -> str:
        """
        :return: Git status
        :rtype: str
        """
        status: str = self.repo.git.status()
        LOGGER.debug('git status: %s', status)
        return status