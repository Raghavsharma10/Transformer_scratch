def addFeatureSA(self, callback, default=None, name=None):
        """
        Add a feature to the suffix array.
        The callback must return a sequence such that
        the feature at position i is attached to the suffix referenced by
        self.SA[i].

        It is called with one argument: the instance of SuffixArray self.
        The callback may traverse self.SA in any fashion.

        The default behavior is to name the new feature after the callback name.
        To give another name, set the argument name accordingly.

        When the feature of an unknown substring of the text is requested,
        the value of the default argument is used.

        If the feature attached to a suffix is independent of the other suffix
        features, then the method addFeature gives a better alternative.

        You may use addFeatureSA as a decorator as in the following example.

        Example: feature named bigram which attach the frequencies of the
        leading bigram to each suffix.

        >>> SA=SuffixArray("mississippi", unit=UNIT_BYTE)

        >>> def bigram(SA):
        ...     res=[0]*SA.length
        ...     end=0
        ...     while end <= SA.length:
        ...
        ...         begin=end-1
        ...         while end < SA.length and  SA._LCP_values[end]>=2:
        ...             if SA.SA[end]+2<=SA.length: #end of string
        ...                 end+=1
        ...
        ...         nbBigram=end-begin
        ...         for i in xrange(begin, end):
        ...             if SA.SA[i]+2<=SA.length:
        ...                 res[i]=nbBigram
        ...
        ...         end+=1
        ...     return res

        >>> SA.addFeatureSA(bigram, 0)

        >>> SA._bigram_values
        [0, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2]

        >>> print str(SA).expandtabs(14) #doctest: +SKIP
        ...     10        'i'           LCP=0 ,       bigram=0
        ...      7        'ippi'        LCP=1 ,       bigram=1
        ...      4        'issippi'     LCP=1 ,       bigram=2
        ...      1        'ississippi'  LCP=4 ,       bigram=2
        ...      0        'mississipp'  LCP=0 ,       bigram=1
        ...      9        'pi'          LCP=0 ,       bigram=1
        ...      8        'ppi'         LCP=1 ,       bigram=1
        ...      6        'sippi'       LCP=0 ,       bigram=2
        ...      3        'sissippi'    LCP=2 ,       bigram=2
        ...      5        'ssippi'      LCP=1 ,       bigram=2
        ...      2        'ssissippi'   LCP=3 ,       bigram=2

        >>> SA.bigram('ip')
        1

        >>> SA.bigram('si')
        2

        >>> SA.bigram('zw')
        0

        """
        if name is None:
            featureName = callback.__name__
        else:
            featureName = name

        featureValues = callback(self)
        setattr(self, "_%s_values" % featureName, featureValues)
        setattr(self, "%s_default" % featureName, default)
        self.features.append(featureName)

        def findFeature(substring):
            res = self._findOne(substring, )
            if res is not False:
                return featureValues[res]
            else:
                return default

        setattr(self, featureName, findFeature)