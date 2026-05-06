def modifiers(self):
        """ For verb phrases (VP), yields a list of the nearest adjectives and adverbs.
        """
        if self._modifiers is None:
            # Iterate over all the chunks and attach modifiers to their VP-anchor.
            is_modifier = lambda ch: ch.type in ("ADJP", "ADVP") and ch.relation is None
            for chunk in self.sentence.chunks:
                chunk._modifiers = []
            for chunk in filter(is_modifier, self.sentence.chunks):
                anchor = chunk.nearest("VP")
                if anchor: anchor._modifiers.append(chunk)
        return self._modifiers