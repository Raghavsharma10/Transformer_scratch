def speaker(self):
        """
        Lazy-loads the speaker for this word

        :getter: Returns the plain string value of the speaker tag for the word
        :type: str

        """
        if self._speaker is None:
            speakers = self._element.xpath('Speaker/text()')
            if len(speakers) > 0:
                self._speaker = speakers[0]
        return self._speaker