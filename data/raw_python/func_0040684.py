def checkout(self, reference: str):
        """
        Checks out a reference.

        If the index is dirty, or if the repository contains untracked files, the function will fail.

        :param reference: reference to check out
        :type reference: str
        """
        LOGGER.info('checking out: %s', reference)
        if not self.index_is_empty():
            LOGGER.error('index contains change; cannot checkout. Status:\n %s', self.status())
            sys.exit(-1)
        if self.is_dirty(untracked=True):
            LOGGER.error('repository is dirty; cannot checkout "%s"', reference)
            LOGGER.error('repository is dirty; cannot checkout. Status:\n %s', self.status())
            sys.exit(-1)

        LOGGER.debug('going through all present references')
        for head in self.repo.heads:
            if head.name == reference:
                LOGGER.debug('resetting repo index and working tree to: %s', reference)
                self.repo.head.reference = head
                self.repo.head.reset(index=True, working_tree=True)
                break
        else:
            LOGGER.error('reference not found: %s', reference)
            sys.exit(-1)