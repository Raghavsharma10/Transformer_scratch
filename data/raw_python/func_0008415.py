def match(self, word):
        """ Return True if the given Word is part of the constraint:
            - the word (or lemma) occurs in Constraint.words, OR
            - the word (or lemma) occurs in Constraint.taxa taxonomy tree, AND
            - the word and/or chunk tags match those defined in the constraint.
            Individual terms in Constraint.words or the taxonomy can contain wildcards (*).
            Some part-of-speech-tags can also contain wildcards: NN*, VB*, JJ*, RB*
            If the given word contains spaces (e.g., proper noun),
            the entire chunk will also be compared.
            For example: Constraint(words=["Mac OS X*"]) 
            matches the word "Mac" if the word occurs in a Chunk("Mac OS X 10.5").
        """
        # If the constraint has a custom function it must return True.
        if self.custom is not None and self.custom(word) is False:
            return False
        # If the constraint can only match the first word, Word.index must be 0.
        if self.first and word.index > 0:
            return False
        # If the constraint defines excluded options, Word can not match any of these.
        if self.exclude and self.exclude.match(word):
            return False
        # If the constraint defines allowed tags, Word.tag needs to match one of these.
        if self.tags:
            if find(lambda w: _match(word.tag, w), self.tags) is None:
                return False
        # If the constraint defines allowed chunks, Word.chunk.tag needs to match one of these.
        if self.chunks:
            ch = word.chunk and word.chunk.tag or None
            if find(lambda w: _match(ch, w), self.chunks) is None:
                return False
        # If the constraint defines allowed role, Word.chunk.tag needs to match one of these.
        if self.roles:
            R = word.chunk and [r2 for r1, r2 in word.chunk.relations] or []
            if find(lambda w: w in R, self.roles) is None:
                return False
        # If the constraint defines allowed words,
        # Word.string.lower() OR Word.lemma needs to match one of these.
        b = True # b==True when word in constraint (or Constraints.words=[]).
        if len(self.words) + len(self.taxa) > 0:
            s1 = word.string.lower()
            s2 = word.lemma
            b = False
            for w in itertools.chain(self.words, self.taxa):
                # If the constraint has a word with spaces (e.g., a proper noun),
                # compare it to the entire chunk.
                try:
                    if " " in w and (s1 in w or s2 and s2 in w or "*" in w):
                        s1 = word.chunk and word.chunk.string.lower() or s1
                        s2 = word.chunk and " ".join([x or "" for x in word.chunk.lemmata]) or s2
                except:
                    s1 = s1
                    s2 = None
                # Compare the word to the allowed words (which can contain wildcards).
                if _match(s1, w):
                    b=True; break
                # Compare the word lemma to the allowed words, e.g.,
                # if "was" is not in the constraint, perhaps "be" is, which is a good match.
                if s2 and _match(s2, w):
                    b=True; break
        # If the constraint defines allowed taxonomy terms,
        # and the given word did not match an allowed word, traverse the taxonomy.
        # The search goes up from the given word to its parents in the taxonomy.
        # This is faster than traversing all the children of terms in Constraint.taxa.
        # The drawback is that:
        # 1) Wildcards in the taxonomy are not detected (use classifiers instead),
        # 2) Classifier.children() has no effect, only Classifier.parent().
        if self.taxa and (not self.words or (self.words and not b)):
            for s in (
              word.string, # "ants"
              word.lemma,  # "ant"
              word.chunk and word.chunk.string or None, # "army ants"
              word.chunk and " ".join([x or "" for x in word.chunk.lemmata]) or None): # "army ant"
                if s is not None:
                    if self.taxonomy.case_sensitive is False:
                        s = s.lower()
                    # Compare ancestors of the word to each term in Constraint.taxa.
                    for p in self.taxonomy.parents(s, recursive=True):
                        if find(lambda s: p==s, self.taxa): # No wildcards.
                            return True
        return b