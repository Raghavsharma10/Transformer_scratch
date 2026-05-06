def value(self, term, **kwargs):
        """ Returns the value of the given term ("many" => "50-200")
        """
        term = self._normalize(term)
        if term in self._values:
            return self._values[term]
        for classifier in self.classifiers:
            v = classifier.value(term, **kwargs)
            if v is not None:
                return v