def train(self, *args, **kwargs):
        """Train the classifier with a labeled feature set and return the
        classifier. Takes the same arguments as the wrapped NLTK class. This
        method is implicitly called when calling ``classify`` or ``accuracy``
        methods and is included only to allow passing in arguments to the
        ``train`` method of the wrapped NLTK class.

        .. versionadded:: 0.6.2

        :rtype: A classifier

        """
        try:
            self.classifier = self.nltk_class.train(self.train_features,
                                                    *args, **kwargs)
            return self.classifier
        except AttributeError:
            raise ValueError("NLTKClassifier must have a nltk_class"
                             " variable that is not None.")