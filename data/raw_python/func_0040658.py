def get_current_branch(self) -> str:
        """
        :return: current branch
        :rtype: str
        """
        current_branch: str = self.repo.active_branch.name
        LOGGER.debug('current branch: %s', current_branch)
        return current_branch