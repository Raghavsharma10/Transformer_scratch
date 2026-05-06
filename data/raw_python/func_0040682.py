def get_sha(self) -> str:
        """
        :return: SHA of the latest commit
        :rtype: str
        """
        current_sha: str = self.repo.head.commit.hexsha
        LOGGER.debug('current commit SHA: %s', current_sha)
        return current_sha