def lemma(self):
        """
        Lazy-loads the lemma for this word

        :getter: Returns the plain string value of the word lemma
        :type: str

        """
        if self._lemma is None:
            lemmata = self._element.xpath('lemma/text()')
            if len(lemmata) > 0:
                self._lemma = lemmata[0]
        return self._lemma