def classify(self, token, previous=None, next=None, **kwargs):
        """ Returns the predicted tag for the given token,
            in context of the given previous and next (token, tag)-tuples.
        """
        return self._classifier.classify(self._v(token, previous, next), **kwargs)