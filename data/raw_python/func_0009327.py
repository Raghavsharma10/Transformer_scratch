def cpe_disjoint(cls, source, target):
        """
        Compares two WFNs and returns True if the set-theoretic relation
        between the names is DISJOINT.

        :param CPE2_3_WFN source: first WFN CPE Name
        :param CPE2_3_WFN target: seconds WFN CPE Name
        :returns: True if the set relation between source and target
            is DISJOINT, otherwise False.
        :rtype: boolean
        """

        # If any pairwise comparison returned DISJOINT  then
        # the overall name relationship is DISJOINT
        for att, result in CPESet2_3.compare_wfns(source, target):
            isDisjoint = result == CPESet2_3.LOGICAL_VALUE_DISJOINT
            if isDisjoint:
                return True
        return False