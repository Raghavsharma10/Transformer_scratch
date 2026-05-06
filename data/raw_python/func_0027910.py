def unstem(self, term):

        """
        Given a stemmed term, get the most common unstemmed variant.

        Args:
            term (str): A stemmed term.

        Returns:
            str: The unstemmed token.
        """

        originals = []
        for i in self.terms[term]:
            originals.append(self.tokens[i]['unstemmed'])

        mode = Counter(originals).most_common(1)
        return mode[0][0]