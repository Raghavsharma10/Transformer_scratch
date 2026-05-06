def name_match(self, wfn):
        """
        Accepts a set of CPE Names K and a candidate CPE Name X. It returns
        'True' if X matches any member of K, and 'False' otherwise.

        :param CPESet self: A set of m known CPE Names K = {K1, K2, …, Km}.
        :param CPE cpe: A candidate CPE Name X.
        :returns: True if X matches K, otherwise False.
        :rtype: boolean
        """

        for N in self.K:
            if CPESet2_3.cpe_superset(wfn, N):
                return True
        return False