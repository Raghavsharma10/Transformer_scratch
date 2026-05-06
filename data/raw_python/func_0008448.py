def noun_phrases(self):
        """Returns a list of noun phrases for this blob."""
        return WordList([phrase.strip()
                         for phrase in self.np_extractor.extract(self.raw)
                         if len(phrase.split()) > 1])