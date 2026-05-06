def match_header(cls, header):
        """A constant value HDU will only be recognized as such if the header
        contains a valid PIXVALUE and NAXIS == 0.
        """

        pixvalue = header.get('PIXVALUE')
        naxis = header.get('NAXIS', 0)

        return (super(_ConstantValueImageBaseHDU, cls).match_header(header) and
                (isinstance(pixvalue, float) or _is_int(pixvalue)) and
                naxis == 0)