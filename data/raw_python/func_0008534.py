def loop(self, *tags):
        """ Iterates over the tags in the entire Sentence,
            For example, Sentence.loop(POS, LEMMA) yields tuples of the part-of-speech tags and lemmata. 
            Possible tags: WORD, LEMMA, POS, CHUNK, PNP, RELATION, ROLE, ANCHOR or a custom word tag.
            Any order or combination of tags can be supplied.
        """
        for i in range(len(self.words)):
            yield tuple([self.get(i, tag=tag) for tag in tags])