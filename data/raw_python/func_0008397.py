def update(self, new_positive_data=None,
               new_unlabeled_data=None, positive_prob_prior=0.5,
               *args, **kwargs):
        '''Update the classifier with new data and re-trains the
        classifier.

        :param new_positive_data: List of new, labeled strings.
        :param new_unlabeled_data: List of new, unlabeled strings.
        '''
        self.positive_prob_prior = positive_prob_prior
        if new_positive_data:
            self.positive_set += new_positive_data
            self.positive_features += [self.extract_features(d)
                                       for d in new_positive_data]
        if new_unlabeled_data:
            self.unlabeled_set += new_unlabeled_data
            self.unlabeled_features += [self.extract_features(d)
                                        for d in new_unlabeled_data]
        self.classifier = self.nltk_class.train(self.positive_features,
                                                self.unlabeled_features,
                                                self.positive_prob_prior,
                                                *args, **kwargs)
        return True