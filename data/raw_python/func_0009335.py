def append(self, cpe):
        """
        Adds a CPE Name to the set if not already.

        :param CPE cpe: CPE Name to store in set
        :returns: None
        :exception: ValueError - invalid version of CPE Name


        TEST:

        >>> from .cpeset2_2 import CPESet2_2
        >>> from .cpe2_2 import CPE2_2
        >>> uri1 = 'cpe:/h:hp'
        >>> c1 = CPE2_2(uri1)
        >>> s = CPESet2_2()
        >>> s.append(c1)
        """

        if cpe.VERSION != CPE.VERSION_2_2:
            errmsg = "CPE Name version {0} not valid, version 2.2 expected".format(
                cpe.VERSION)
            raise ValueError(errmsg)

        for k in self.K:
            if cpe.cpe_str == k.cpe_str:
                return None

        self.K.append(cpe)