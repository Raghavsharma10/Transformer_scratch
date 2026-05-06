def word(self):
        """
        Lazy-loads word value

        :getter: Returns the plain string value of the word
        :type: str

        """
        if self._word is None:
            words = self._element.xpath('word/text()')
            if len(words) > 0:
                self._word = words[0]
        return self._word