def character_offset_begin(self):
        """
        Lazy-loads character offset begin node

        :getter: Returns the integer value of the beginning offset
        :type: int

        """
        if self._character_offset_begin is None:
            offsets = self._element.xpath('CharacterOffsetBegin/text()')
            if len(offsets) > 0:
                self._character_offset_begin = int(offsets[0])
        return self._character_offset_begin