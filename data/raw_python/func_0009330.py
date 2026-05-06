def cpe_superset(cls, source, target):
        """
        Compares two WFNs and returns True if the set-theoretic relation
        between the names is (non-proper) SUPERSET.

        :param CPE2_3_WFN source: first WFN CPE Name
        :param CPE2_3_WFN target: seconds WFN CPE Name
        :returns: True if the set relation between source and target
            is SUPERSET, otherwise False.
        :rtype: boolean
        """

        # If any pairwise comparison returned something other than SUPERSET
        # or EQUAL, then SUPERSET is False.
        for att, result in CPESet2_3.compare_wfns(source, target):
            isSuperset = result == CPESet2_3.LOGICAL_VALUE_SUPERSET
            isEqual = result == CPESet2_3.LOGICAL_VALUE_EQUAL
            if (not isSuperset) and (not isEqual):
                return False

        return True