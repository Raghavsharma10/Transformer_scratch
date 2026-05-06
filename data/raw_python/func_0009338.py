def name_match(self, cpe):
        """
        Accepts a set of known instances of CPE Names and a candidate CPE Name,
        and returns 'True' if the candidate can be shown to be
        an instance based on the content of the known instances.
        Otherwise, it returns 'False'.

        :param CPESet self: A set of m known CPE Names K = {K1, K2, …, Km}.
        :param CPE cpe: A candidate CPE Name X.
        :returns: True if X matches K, otherwise False.
        :rtype: boolean

        TEST: matching with identical CPE in set

        >>> from .cpe1_1 import CPE1_1
        >>> from .cpeset1_1 import CPESet1_1
        >>> uri1 = 'cpe://microsoft:windows:xp!vista'
        >>> uri2 = 'cpe:/cisco::3825;cisco:2:44/cisco:ios:12.3:enterprise'
        >>> c1 = CPE1_1(uri1)
        >>> c2 = CPE1_1(uri2)
        >>> s = CPESet1_1()
        >>> s.append(c1)
        >>> s.append(c2)
        >>> s.name_match(c2)
        True

        """

        # An empty set not matching with any CPE
        if len(self) == 0:
            return False

        # If input CPE Name string is in set of CPE Name strings
        # not do searching more because there is a matching
        for k in self.K:
            if (k.cpe_str == cpe.cpe_str):
                return True

        # There are not a CPE Name string in set equal to
        # input CPE Name string
        match = False

        for p in CPE.CPE_PART_KEYS:
            elems_cpe = cpe.get(p)
            for ec in elems_cpe:
                # Search of element of part of input CPE

                # Each element ec of input cpe[p] is compared with
                # each element ek of k[p] in set K

                for k in self.K:
                    elems_k = k.get(p)

                    for ek in elems_k:
                        # Matching

                        # Each component in element ec is compared with
                        # each component in element ek
                        for ck in CPEComponent.CPE_COMP_KEYS:
                            comp_cpe = ec.get(ck)
                            comp_k = ek.get(ck)

                            match = comp_k in comp_cpe

                            if not match:
                                # Search compoment in another element ek[p]
                                break

                            # Component analyzed

                        if match:
                            # Element matched
                            break
                    if match:
                        break
                # Next element in part in "cpe"

                if not match:
                    # cpe part not match with parts in set
                    return False

            # Next part in input CPE Name

        # All parts in input CPE Name matched
        return True