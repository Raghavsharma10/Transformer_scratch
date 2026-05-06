def _findOne(self, subString):
        """
        >>> SA=SuffixArray("mississippi", unit=UNIT_BYTE)
        >>> SA._findOne("ippi")
        1

        >>> SA._findOne("missi")
        4
        """
        SA = self.SA
        LCPs = self._LCP_values
        string = self.string

        try:
            subString = _array("i", [self.tokId[c] for c in self.tokenize(subString)])
        except KeyError:
            # if a token of the subString is not in the vocabulary
            # the substring can't be in the string
            return False
        lenSubString = len(subString)

        #################################
        # Dichotomy search of subString #
        #################################
        lower = 0
        upper = self.length
        success = False

        while upper - lower > 0:
            middle = (lower + upper) // 2

            middleSubString = string[SA[middle]:min(SA[middle] + lenSubString, self.length)]

            # NOTE: the cmp function is removed in Python 3
            # Strictly speaking we are doing one comparison more now
            if subString < middleSubString:
                upper = middle
            elif subString > middleSubString:
                lower = middle + 1
            else:
                success = True
                break

        if not success:
            return False
        else:
            return middle