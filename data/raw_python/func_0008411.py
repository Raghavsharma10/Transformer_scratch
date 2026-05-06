def classify(self, term, **kwargs):
        """ Returns the (most recently added) semantic type for the given term ("many" => "quantity").
            If the term is not in the dictionary, try Taxonomy.classifiers.
        """
        term = self._normalize(term)
        if dict.__contains__(self, term):
            return self[term][0].keys()[-1]
        # If the term is not in the dictionary, check the classifiers.
        # Returns the first term in the list returned by a classifier.
        for classifier in self.classifiers:
            # **kwargs are useful if the classifier requests extra information,
            # for example the part-of-speech tag.
            v = classifier.parents(term, **kwargs)
            if v:
                return v[0]