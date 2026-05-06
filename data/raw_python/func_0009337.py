def append(self, cpe):
        """
        Adds a CPE Name to the set if not already.

        :param CPE cpe: CPE Name to store in set
        :returns: None
        :exception: ValueError - invalid version of CPE Name

        TEST:

        >>> from .cpeset1_1 import CPESet1_1
        >>> from .cpe1_1 import CPE1_1
        >>> uri1 = 'cpe://microsoft:windows:xp!vista'
        >>> c1 = CPE1_1(uri1)
        >>> s = CPESet1_1()
        >>> s.append(c1)
        """

        if cpe.VERSION != CPE.VERSION_1_1:
            msg = "CPE Name version {0} not valid, version 1.1 expected".format(
                cpe.VERSION)
            raise ValueError(msg)

        for k in self.K:
            if cpe.cpe_str == k.cpe_str:
                return None

        self.K.append(cpe)