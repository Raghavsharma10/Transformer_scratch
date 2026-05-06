def _process_tags(self, tags):
        """Process the tags of the message

        :param tags: the tags string of a message
        :type tags: :class:`str` | None
        :returns: list of tags
        :rtype: :class:`list` of :class:`message.Tag`
        :raises: None
        """
        if not tags:
            return []
        return [message.Tag.from_str(x) for x in tags.split(';')]