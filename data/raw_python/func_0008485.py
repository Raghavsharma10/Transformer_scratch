def find_chunks(self, tokens, **kwargs):
        """ Annotates the given list of tokens with chunk tags.
            Several tags can be added, for example chunk + preposition tags.
        """
        # [["The", "DT"], ["cat", "NN"], ["purs", "VB"]] =>
        # [["The", "DT", "B-NP"], ["cat", "NN", "I-NP"], ["purs", "VB", "B-VP"]]
        return find_prepositions(
               find_chunks(tokens,
                   language = kwargs.get("language", self.language)))