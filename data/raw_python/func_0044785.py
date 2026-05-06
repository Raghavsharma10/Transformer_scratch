def pos(self):
        """
        Lazy-loads the part of speech tag for this word

        :getter: Returns the plain string value of the POS tag for the word
        :type: str

        """
        if self._pos is None:
            poses = self._element.xpath('POS/text()')
            if len(poses) > 0:
                self._pos = poses[0]
        return self._pos