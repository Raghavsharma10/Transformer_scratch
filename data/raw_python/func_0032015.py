def from_str(cls, tagstring):
        """Create a tag by parsing the tag of a message

        :param tagstring: A tag string described in the irc protocol
        :type tagstring: :class:`str`
        :returns: A tag
        :rtype: :class:`Tag`
        :raises: None
        """
        m = cls._parse_regexp.match(tagstring)
        return cls(name=m.group('name'), value=m.group('value'), vendor=m.group('vendor'))