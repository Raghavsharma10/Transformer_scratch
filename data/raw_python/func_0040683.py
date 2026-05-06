def get_short_sha(self) -> str:
        """
        :return: short SHA of the latest commit
        :rtype: str
        """
        short_sha: str = self.get_sha()[:7]
        LOGGER.debug('short SHA: %s', short_sha)
        return short_sha