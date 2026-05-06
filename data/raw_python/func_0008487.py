def split(self, sep=TOKENS):
        """ Returns a list of sentences, where each sentence is a list of tokens,
            where each token is a list of word + tags.
        """
        if sep != TOKENS:
            return unicode.split(self, sep)
        if len(self) == 0:
            return []
        return [[[x.replace("&slash;", "/") for x in token.split("/")]
            for token in sentence.split(" ")]
                for sentence in unicode.split(self, "\n")]