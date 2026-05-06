def find(self, subString, features=[]):
        """
        Dichotomy search of subString in the suffix array.
        As soon as a suffix which starts with subString is found,
        it uses the LCPs in order to find the other matching suffixes.

        The outputs consists in a list of tuple (pos, feature0, feature1, ...)
        where feature0, feature1, ... are the features attached to the suffix
        at position pos.
        Features are listed in the same order as requested in the input list of
        features [featureName0, featureName1, ...]

        >>> SA=SuffixArray('mississippi', UNIT_BYTE)
        >>> SA.find("ssi")
        array('i', [5, 2])

        >>> SA.find("mi")
        array('i', [0])

        >>> SA=SuffixArray('miss A and miss B', UNIT_WORD)
        >>> SA.find("miss")
        array('i', [0, 3])

        >>> SA=SuffixArray('mississippi', UNIT_BYTE)
        >>> SA.find("iss", ['LCP'])
        [(4, 1), (1, 4)]

        >>> SA=SuffixArray('mississippi', UNIT_BYTE)
        >>> SA.find("A")
        array('i')

        >>> SA=SuffixArray('mississippi', UNIT_BYTE)
        >>> SA.find("pp")
        array('i', [8])

        >>> SA=SuffixArray('mississippi', UNIT_BYTE)
        >>> SA.find("ppp")
        array('i')


        >>> SA=SuffixArray('mississippi', UNIT_BYTE)
        >>> SA.find("im")
        array('i')
        """
        SA = self.SA
        LCPs = self._LCP_values
        string = self.string

        middle = self._findOne(subString)
        if middle is False:
            return _array('i')

        subString = _array("i", [self.tokId[c] for c in self.tokenize(subString)])
        lenSubString = len(subString)

        ###########################################
        # Use LCPS to retrieve the other suffixes #
        ###########################################
        lower = middle
        upper = middle + 1
        middleLCP = LCPs[middle]
        while lower > 0 and LCPs[lower] >= lenSubString:
            lower -= 1

        while upper < self.length and LCPs[upper] >= lenSubString:
            upper += 1

        ###############################################
        # When features is empty, outputs a flat list #
        ###############################################
        res = SA[lower:upper]
        if len(features) == 0:
            return res

        ##############################################
        # When features is non empty, outputs a list #
        # of tuples (pos, feature_1, feature_2, ...) #
        ##############################################
        else:
            features = [getattr(self, "_%s_values" % featureName) for featureName in features]
            features = [featureValues[lower:upper] for featureValues in features]

            return zip(res, *features)