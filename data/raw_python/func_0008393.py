def accuracy(self, test_set, format=None):
        """Compute the accuracy on a test set.

        :param test_set: A list of tuples of the form ``(text, label)``, or a
            filename.
        :param format: If ``test_set`` is a filename, the file format, e.g.
            ``"csv"`` or ``"json"``. If ``None``, will attempt to detect the
            file format.

        """
        if isinstance(test_set, basestring):  # test_set is a filename
            test_data = self._read_data(test_set)
        else:  # test_set is a list of tuples
            test_data = test_set
        test_features = [(self.extract_features(d), c) for d, c in test_data]
        return nltk.classify.accuracy(self.classifier, test_features)