def ner(self):
        """
        Lazy-loads the NER for this word

        :getter: Returns the plain string value of the NER tag for the word
        :type: str

        """
        if self._ner is None:
            ners = self._element.xpath('NER/text()')
            if len(ners) > 0:
                self._ner = ners[0]
        return self._ner