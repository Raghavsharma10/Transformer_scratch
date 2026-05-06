def from_xsc(cls, path):
        """ Returns u.Quantity with box vectors from XSC file """

        def parse(path):
            """
            Open and parses an XSC file into its fields

            Parameters
            ----------
            path : str
                Path to XSC file

            Returns
            -------
            namedxsc : namedtuple
                A namedtuple with XSC fields as names
            """
            with open(path) as f:
                lines = f.readlines()
            NamedXsc = namedtuple('NamedXsc', lines[1].split()[1:])
            return NamedXsc(*map(float, lines[2].split()))

        xsc = parse(path)
        return u.Quantity([[xsc.a_x, xsc.a_y, xsc.a_z],
                           [xsc.b_x, xsc.b_y, xsc.b_z],
                           [xsc.c_x, xsc.c_y, xsc.c_z]], unit=u.angstroms)