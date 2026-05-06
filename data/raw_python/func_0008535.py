def indexof(self, value, tag=WORD):
        """ Returns the indices of tokens in the sentence where the given token tag equals the string.
            The string can contain a wildcard "*" at the end (this way "NN*" will match "NN" and "NNS").
            The tag can be WORD, LEMMA, POS, CHUNK, PNP, RELATION, ROLE, ANCHOR or a custom word tag.
            For example: Sentence.indexof("VP", tag=CHUNK) 
            returns the indices of all the words that are part of a VP chunk.
        """
        match = lambda a, b: a.endswith("*") and b.startswith(a[:-1]) or a==b
        indices = []
        for i in range(len(self.words)):
            if match(value, unicode(self.get(i, tag))):
                indices.append(i)
        return indices