def append(self, cpe):
        """
        Adds a CPE element to the set if not already.
        Only WFN CPE Names are valid, so this function converts the input CPE
        object of version 2.3 to WFN style.

        :param CPE cpe: CPE Name to store in set
        :returns: None
        :exception: ValueError - invalid version of CPE Name
        """

        if cpe.VERSION != CPE2_3.VERSION:
            errmsg = "CPE Name version {0} not valid, version 2.3 expected".format(
                cpe.VERSION)
            raise ValueError(errmsg)

        for k in self.K:
            if cpe._str == k._str:
                return None

        if isinstance(cpe, CPE2_3_WFN):
            self.K.append(cpe)
        else:
            # Convert the CPE Name to WFN
            wfn = CPE2_3_WFN(cpe.as_wfn())
            self.K.append(wfn)