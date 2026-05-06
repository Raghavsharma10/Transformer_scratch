def prob_classify(self, text):
        """Return the label probability distribution for classifying a string
        of text.

        Example:
        ::

            >>> classifier = NaiveBayesClassifier(train_data)
            >>> prob_dist = classifier.prob_classify("I feel happy this morning.")
            >>> prob_dist.max()
            'positive'
            >>> prob_dist.prob("positive")
            0.7

        :rtype: nltk.probability.DictionaryProbDist

        """
        text_features = self.extract_features(text)
        return self.classifier.prob_classify(text_features)