def append(self, word, lemma=None, type=None, chunk=None, role=None, relation=None, pnp=None, anchor=None, iob=None, custom={}):
        """ Appends the next word to the sentence / chunk / preposition.
            For example: Sentence.append("clawed", "claw", "VB", "VP", role=None, relation=1)
            - word     : the current word,
            - lemma    : the canonical form of the word,
            - type     : part-of-speech tag for the word (NN, JJ, ...),
            - chunk    : part-of-speech tag for the chunk this word is part of (NP, VP, ...),
            - role     : the chunk's grammatical role (SBJ, OBJ, ...),
            - relation : an id shared by other related chunks (e.g., SBJ-1 <=> VP-1),
            - pnp      : PNP if this word is in a prepositional noun phrase (B- prefix optional),
            - iob      : BEGIN if the word marks the start of a new chunk,
                         INSIDE (optional) if the word is part of the previous chunk,
            - custom   : a dictionary of (tag, value)-items for user-defined word tags.
        """
        self._do_word(word, lemma, type)            # Append Word object.
        self._do_chunk(chunk, role, relation, iob)  # Append Chunk, or add last word to last chunk.
        self._do_conjunction()
        self._do_relation()
        self._do_pnp(pnp, anchor)
        self._do_anchor(anchor)
        self._do_custom(custom)