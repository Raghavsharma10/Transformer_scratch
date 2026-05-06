def tag(self, tag: str, overwrite: bool = False) -> None:
        """
        Tags the current commit

        :param tag: tag
        :type tag: str
        :param overwrite: overwrite existing tag
        :type overwrite: bool
        """
        LOGGER.info('tagging repo: %s', tag)
        try:
            self.repo.create_tag(tag)
        except GitCommandError as exc:
            if 'already exists' in exc.stderr and overwrite:
                LOGGER.info('overwriting existing tag')
                self.remove_tag(tag)
                self.repo.create_tag(tag)
            else:
                LOGGER.exception('error while tagging repo')
                raise