def fromFile(cls, filename):
        """
        Load a suffix array instance from filename, a file created by
        toFile.
        Accept any filename following the _open conventions.
        """
        self = cls.__new__(cls)  # new instance which does not call __init__

        start = _time()

        savedData = _loads(_open(filename, "r").read())

        # load common attributes
        self.string, self.unit, self.voc, self.vocSize, self.SA, features = savedData[:6]
        self.length = len(self.SA)

        # determine token delimiter
        if self.unit == UNIT_WORD:
            self.tokSep = " "
        elif self.unit in (UNIT_CHARACTER, UNIT_BYTE):
            self.tokSep = ""
        else:
            raise Exception("Unknown unit type identifier:", self.unit)

        # recompute tokId based on voc
        self.tokId = dict((char, iChar) for iChar, char in enumerate(self.voc))
        self.nbSentences = self.string.count(self.tokId.get("\n", 0))

        # Load features
        self.features = []
        for featureName, (featureValues, featureDefault) in zip(features, savedData[6:]):
            self.addFeatureSA((lambda _: featureValues), name=featureName, default=featureDefault)

        self.fromFileTime = _time() - start
        if _trace: print >> _stderr, "fromFileTime %.2fs" % self.fromFileTime
        return self