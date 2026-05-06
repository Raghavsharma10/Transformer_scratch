def emotes(self, emotes):
        """Set the emotes

        :param emotes: the key of the emotes tag
        :type emotes: :class:`str`
        :returns: None
        :rtype: None
        :raises: None
        """
        if emotes is None:
            self._emotes = []
            return
        es = []
        for estr in emotes.split('/'):
            es.append(Emote.from_str(estr))
        self._emotes = es