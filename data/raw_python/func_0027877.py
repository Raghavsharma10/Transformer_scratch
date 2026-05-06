def index(self, text, terms=None, **kwargs):

        """
        Index all term pair distances.

        Args:
            text (Text): The source text.
            terms (list): Terms to index.
        """

        self.clear()

        # By default, use all terms.
        terms = terms or text.terms.keys()

        pairs = combinations(terms, 2)
        count = comb(len(terms), 2)

        for t1, t2 in bar(pairs, expected_size=count, every=1000):

            # Set the Bray-Curtis distance.
            score = text.score_braycurtis(t1, t2, **kwargs)
            self.set_pair(t1, t2, score)