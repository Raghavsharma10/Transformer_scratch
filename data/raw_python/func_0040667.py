def last_commit_msg(self) -> str:
        """
        :return: last commit message
        :rtype: str
        """
        last_msg: str = self.latest_commit().message.rstrip()
        LOGGER.debug('last msg: %s', last_msg)
        return last_msg