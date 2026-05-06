def as_wfn(self):
        r"""
        Returns the value of compoment encoded as Well-Formed Name (WFN)
        string.

        :returns: WFN string
        :rtype: string

        TEST:

        >>> val = 'xp!vista'
        >>> comp1 = CPEComponent1_1(val, CPEComponentSimple.ATT_VERSION)
        >>> comp1.as_wfn()
        'xp\\!vista'
        """

        result = []

        for s in self._standard_value:
            result.append(s)
            result.append(CPEComponent1_1._ESCAPE_SEPARATOR)

        return "".join(result[0:-1])