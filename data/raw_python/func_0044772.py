def sentiment(self):
        """
        The sentiment of this sentence

        :getter: Returns the sentiment value of this sentence
        :type: int

        """
        if self._sentiment is None:
            self._sentiment = int(self._element.get('sentiment'))
        return self._sentiment