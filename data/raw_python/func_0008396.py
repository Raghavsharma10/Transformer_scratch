def train(self, *args, **kwargs):
        """Train the classifier with a labeled and unlabeled feature sets and
        return the classifier. Takes the same arguments as the wrapped NLTK
        class. This method is implicitly called when calling ``classify`` or
        ``accuracy`` methods and is included only to allow passing in arguments
        to the ``train`` method of the wrapped NLTK class.

        :rtype: A classifier

        """
        self.classifier = self.nltk_class.train(self.positive_features,
                                                self.unlabeled_features,
                                                self.positive_prob_prior)
        return self.classifier