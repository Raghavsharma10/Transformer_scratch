def is_on_tag(self) -> bool:
        """
        :return: True if latest commit is tagged
        :rtype: bool
        """
        if self.get_current_tag():
            LOGGER.debug('latest commit is tagged')
            return True

        LOGGER.debug('latest commit is NOT tagged')
        return False