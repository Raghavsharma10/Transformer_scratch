def cpe_equal(cls, source, target):
        """
        Compares two WFNs and returns True if the set-theoretic relation
        between the names is EQUAL.

        :param CPE2_3_WFN source: first WFN CPE Name
        :param CPE2_3_WFN target: seconds WFN CPE Name
        :returns: True if the set relation between source and target
            is EQUAL, otherwise False.
        :rtype: boolean
        """

        # If any pairwise comparison returned EQUAL then
        # the overall name relationship is EQUAL
        for att, result in CPESet2_3.compare_wfns(source, target):
            isEqual = result == CPESet2_3.LOGICAL_VALUE_EQUAL
            if not isEqual:
                return False
        return True