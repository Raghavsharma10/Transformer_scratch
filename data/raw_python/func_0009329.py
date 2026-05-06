def cpe_subset(cls, source, target):
        """
        Compares two WFNs and returns True if the set-theoretic relation
        between the names is (non-proper) SUBSET.

        :param CPE2_3_WFN source: first WFN CPE Name
        :param CPE2_3_WFN target: seconds WFN CPE Name
        :returns: True if the set relation between source and target
            is SUBSET, otherwise False.
        :rtype: boolean
        """

        # If any pairwise comparison returned something other than SUBSET
        # or EQUAL, then SUBSET is False.
        for att, result in CPESet2_3.compare_wfns(source, target):
            isSubset = result == CPESet2_3.LOGICAL_VALUE_SUBSET
            isEqual = result == CPESet2_3.LOGICAL_VALUE_EQUAL
            if (not isSubset) and (not isEqual):
                return False
        return True