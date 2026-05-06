def sent_tokenize(self, text, **kwargs):
        """NLTK's sentence tokenizer (currently PunktSentenceTokenizer).

        Uses an unsupervised algorithm to build a model for abbreviation
        words, collocations, and words that start sentences, then uses
        that to find sentence boundaries.

        """
        sentences = self.sent_tok.tokenize(
            text,
            realign_boundaries=kwargs.get(
                "realign_boundaries",
                True))
        return sentences