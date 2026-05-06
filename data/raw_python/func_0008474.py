def apply(self, token, previous=(None, None), next=(None, None)):
        """ Returns a (token, tag)-tuple for the given token,
            in context of the given previous and next (token, tag)-tuples.
        """
        return [token[0], self._classifier.classify(self._v(token[0], previous, next))]