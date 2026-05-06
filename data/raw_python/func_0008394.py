def update(self, new_data, *args, **kwargs):
        '''Update the classifier with new training data and re-trains the
        classifier.

        :param new_data: New data as a list of tuples of the form
            ``(text, label)``.
        '''
        self.train_set += new_data
        self.train_features = [(self.extract_features(d), c)
                               for d, c in self.train_set]
        try:
            self.classifier = self.nltk_class.train(self.train_features,
                                                    *args, **kwargs)
        except AttributeError:  # Descendant has not defined nltk_class
            raise ValueError("NLTKClassifier must have a nltk_class"
                             " variable that is not None.")
        return True