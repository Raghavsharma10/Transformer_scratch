def character_offset_end(self):
        """
        Lazy-loads character offset end node

        :getter: Returns the integer value of the ending offset
        :type: int

        """
        if self._character_offset_end is None:
            offsets = self._element.xpath('CharacterOffsetEnd/text()')
            if len(offsets) > 0:
                self._character_offset_end = int(offsets[0])
        return self._character_offset_end