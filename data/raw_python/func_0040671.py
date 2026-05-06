def index_is_empty(self) -> bool:
        """
        :return: True if index is empty (no staged changes)
        :rtype: bool
        """
        index_empty: bool = len(self.repo.index.diff(self.repo.head.commit)) == 0
        LOGGER.debug('index is empty: %s', index_empty)
        return index_empty