def synset(self, id, pos=ADJECTIVE):
        """ Returns a (polarity, subjectivity)-tuple for the given synset id.
            For example, the adjective "horrible" has id 193480 in WordNet:
            Sentiment.synset(193480, pos="JJ") => (-0.6, 1.0, 1.0).
        """
        id = str(id).zfill(8)
        if not id.startswith(("n-", "v-", "a-", "r-")):
            if pos == NOUN:
                id = "n-" + id
            if pos == VERB:
                id = "v-" + id
            if pos == ADJECTIVE:
                id = "a-" + id
            if pos == ADVERB:
                id = "r-" + id
        if dict.__len__(self) == 0:
            self.load()
        try:
            return tuple(self._synsets[id])[:2]
        except KeyError: # Some WordNet id's are not zero padded.
            return tuple(self._synsets.get(re.sub(r"-0+", "-", id), (0.0, 0.0))[:2])