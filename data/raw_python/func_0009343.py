def _fact_ref_eval(cls, cpeset, wfn):
        """
        Returns True if wfn is a non-proper superset (True superset
        or equal to) any of the names in cpeset, otherwise False.

        :param CPESet cpeset: list of CPE bound Names.
        :param CPE2_3_WFN wfn: WFN CPE Name.
        :returns: True if wfn is a non-proper superset any of the names in cpeset, otherwise False
        :rtype: boolean
        """

        for n in cpeset:
            # Need to convert each n from bound form to WFN
            if (CPESet2_3.cpe_superset(wfn, n)):
                return True

        return False