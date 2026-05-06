def get_current_tag(self) -> typing.Optional[str]:
        """
        :return: tag name if current commit is on tag, else None
        :rtype: optional str
        """
        tags = list(self.repo.tags)
        if not tags:
            LOGGER.debug('no tag found')
            return None
        for tag in tags:
            LOGGER.debug('tag found: %s; comparing with commit', tag)
            if tag.commit == self.latest_commit():
                tag_name: str = tag.name
                LOGGER.debug('found tag on commit: %s', tag_name)
                return tag_name

        LOGGER.debug('no tag found on latest commit')
        return None