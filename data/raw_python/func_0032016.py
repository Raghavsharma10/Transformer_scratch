def from_str(cls, emotestr):
        """Create an emote from the emote tag key

        :param emotestr: the tag key, e.g. ``'123:0-4'``
        :type emotestr: :class:`str`
        :returns: an emote
        :rtype: :class:`Emote`
        :raises: None
        """
        emoteid, occstr = emotestr.split(':')
        occurences = []
        for occ in occstr.split(','):
            start, end = occ.split('-')
            occurences.append((int(start), int(end)))
        return cls(int(emoteid), occurences)