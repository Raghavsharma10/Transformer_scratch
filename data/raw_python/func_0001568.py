def addFeature(self, callback, default=None, name=None, arguments=None):
        """
        Add a feature to the suffix array.
        The callback must return the feature corresponding to the suffix at
        position self.SA[i].

        The callback must be callable (a function or lambda).
        The argument names of the callback are used to determine the data
        needed. If an argument is the name of feature already defined, then
        this argument will be the value of that feature for the current suffix.
        In addition the argument pos is the position of the current suffix
        and iSA is the index of pos in SA.
        Other attributes of the SuffixArray instance may be use as argument
        names.

        If the feature attached to a suffix depends on other suffix features,
        then the method addFeatureSA is the only choice.

        """
        if name is None:
            featureName = callback.__name__
        else:
            featureName = name

        if arguments is None:
            signature = getargspec(callback)[0]
        else:
            signature = arguments

        featureValues = [default] * (self.length)
        args = [getattr(self, "_%s_values" % featName) for featName in signature]
        # print args
        for i, pos in enumerate(self.SA):
            arg = [j[i] for j in args]
            # print arg
            featureValues[i] = callback(*arg)
        # end alternative

        setattr(self, "_%s_values" % featureName, featureValues)
        setattr(self, "%s_default" % featureName, default)
        self.features.append(featureName)

        def findFeature(substring):
            res = self._findOne(substring)
            if res:
                return featureValues[res]
            else:
                return default

        setattr(self, featureName, findFeature)