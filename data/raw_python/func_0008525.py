def parse_token(self, token, tags=[WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA]):
        """ Returns the arguments for Sentence.append() from a tagged token representation.
            The order in which token tags appear can be specified.
            The default order is (separated by slashes): 
            - word, 
            - part-of-speech, 
            - (IOB-)chunk, 
            - (IOB-)preposition, 
            - chunk(-relation)(-role), 
            - anchor, 
            - lemma.
            Examples:
            The/DT/B-NP/O/NP-SBJ-1/O/the
            cats/NNS/I-NP/O/NP-SBJ-1/O/cat
            clawed/VBD/B-VP/O/VP-1/A1/claw
            at/IN/B-PP/B-PNP/PP/P1/at
            the/DT/B-NP/I-PNP/NP/P1/the
            sofa/NN/I-NP/I-PNP/NP/P1/sofa
            ././O/O/O/O/.
            Returns a (word, lemma, type, chunk, role, relation, preposition, anchor, iob, custom)-tuple,
            which can be passed to Sentence.append(): Sentence.append(*Sentence.parse_token("cats/NNS/NP"))
            The custom value is a dictionary of (tag, value)-items of unrecognized tags in the token.
        """
        p = { WORD: "", 
               POS: None, 
               IOB: None,
             CHUNK: None,
               PNP: None,
               REL: None,
              ROLE: None,
            ANCHOR: None,
             LEMMA: None }
        # Split the slash-formatted token into separate tags in the given order.
        # Decode &slash; characters (usually in words and lemmata).
        # Assume None for missing tags (except the word itself, which defaults to an empty string).
        custom = {}
        for k, v in izip(tags, token.split("/")):
            if SLASH0 in v:
                v = v.replace(SLASH, "/")
            if k not in p:
                custom[k] = None
            if v != OUTSIDE or k == WORD or k == LEMMA: # "type O negative" => "O" != OUTSIDE.
                (p if k not in custom else custom)[k] = v
        # Split IOB-prefix from the chunk tag:
        # B- marks the start of a new chunk, 
        # I- marks inside of a chunk.
        ch = p[CHUNK]
        if ch is not None and ch.startswith(("B-", "I-")):
            p[IOB], p[CHUNK] = ch[:1], ch[2:] # B-NP
        # Split the role from the relation:
        # NP-SBJ-1 => relation id is 1 and role is SBJ, 
        # VP-1 => relation id is 1 with no role.
        # Tokens may be tagged with multiple relations (e.g., NP-OBJ-1*NP-OBJ-3).
        if p[REL] is not None:
            ch, p[REL], p[ROLE] = self._parse_relation(p[REL])
            # Infer a missing chunk tag from the relation tag (e.g., NP-SBJ-1 => NP).
            # For PP relation tags (e.g., PP-CLR-1), the first chunk is PP, the following chunks NP.
            if ch == "PP" \
             and self._previous \
             and self._previous[REL] == p[REL] \
             and self._previous[ROLE] == p[ROLE]: 
                ch = "NP"
            if p[CHUNK] is None and ch != OUTSIDE:
                p[CHUNK] = ch
        self._previous = p
        # Return the tags in the right order for Sentence.append().
        return p[WORD], p[LEMMA], p[POS], p[CHUNK], p[ROLE], p[REL], p[PNP], p[ANCHOR], p[IOB], custom