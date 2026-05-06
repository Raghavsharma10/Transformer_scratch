def train(self, token, tag, previous=None, next=None):
        """ Trains the model to predict the given tag for the given token,
            in context of the given previous and next (token, tag)-tuples.
        """
        self._classifier.train(self._v(token, previous, next), type=tag)