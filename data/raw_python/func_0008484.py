def find_tags(self, tokens, **kwargs):
        """ Annotates the given list of tokens with part-of-speech tags.
            Returns a list of tokens, where each token is now a [word, tag]-list.
        """
        # ["The", "cat", "purs"] => [["The", "DT"], ["cat", "NN"], ["purs", "VB"]]
        return find_tags(tokens,
                    lexicon = kwargs.get(   "lexicon", self.lexicon or {}),
                      model = kwargs.get(     "model", self.model),
                 morphology = kwargs.get("morphology", self.morphology),
                    context = kwargs.get(   "context", self.context),
                   entities = kwargs.get(  "entities", self.entities),
                   language = kwargs.get(  "language", self.language),
                    default = kwargs.get(   "default", self.default),
                        map = kwargs.get(       "map", None))