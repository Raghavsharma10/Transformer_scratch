def list_tags(self, pattern: str = None) -> typing.List[str]:
        """
        Returns list of tags, optionally matching "pattern"

        :param pattern: optional pattern to filter results
        :type pattern: str
        :return: existing tags
        :rtype: list of str
        """
        tags: typing.List[str] = [str(tag) for tag in self.repo.tags]
        if not pattern:
            LOGGER.debug('tags found in repo: %s', tags)
            return tags

        LOGGER.debug('filtering tags with pattern: %s', pattern)
        filtered_tags: typing.List[str] = [tag for tag in tags if pattern in tag]
        LOGGER.debug('filtered tags: %s', filtered_tags)
        return filtered_tags