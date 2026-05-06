def slice(self, start, stop):
        """ Returns a portion of the sentence from word start index to word stop index.
            The returned slice is a subclass of Sentence and a deep copy.
        """
        s = Slice(token=self.token, language=self.language)
        for i, word in enumerate(self.words[start:stop]):
            # The easiest way to copy (part of) a sentence
            # is by unpacking all of the token tags and passing them to Sentence.append().
            p0 = word.string                                                       # WORD
            p1 = word.lemma                                                        # LEMMA
            p2 = word.type                                                         # POS
            p3 = word.chunk is not None and word.chunk.type or None                # CHUNK
            p4 = word.pnp is not None and "PNP" or None                            # PNP
            p5 = word.chunk is not None and unzip(0, word.chunk.relations) or None # REL            
            p6 = word.chunk is not None and unzip(1, word.chunk.relations) or None # ROLE
            p7 = word.chunk and word.chunk.anchor_id or None                       # ANCHOR
            p8 = word.chunk and word.chunk.start == start+i and BEGIN or None      # IOB
            p9 = word.custom_tags                                                  # User-defined tags.
            # If the given range does not contain the chunk head, remove the chunk tags.
            if word.chunk is not None and (word.chunk.stop > stop):
                p3, p4, p5, p6, p7, p8 = None, None, None, None, None, None
            # If the word starts the preposition, add the IOB B-prefix (i.e., B-PNP).
            if word.pnp is not None and word.pnp.start == start+i:
                p4 = BEGIN+"-"+"PNP"
            # If the given range does not contain the entire PNP, remove the PNP tags.
            # The range must contain the entire PNP, 
            # since it starts with the PP and ends with the chunk head (and is meaningless without these).
            if word.pnp is not None and (word.pnp.start < start or word.chunk.stop > stop):
                p4, p7 = None, None
            s.append(word=p0, lemma=p1, type=p2, chunk=p3, pnp=p4, relation=p5, role=p6, anchor=p7, iob=p8, custom=p9)
        s.parent = self
        s._start = start
        return s