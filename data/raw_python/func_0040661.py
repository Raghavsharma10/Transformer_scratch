def remove_tag(self, *tag: str):
        """
        Removes tag(s) from the rpo

        :param tag: tags to remove
        :type tag: tuple
        """
        LOGGER.info('removing tag(s) from repo: %s', tag)

        self.repo.delete_tag(*tag)