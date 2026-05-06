def hasMZSignature(self, rd): 
        """
        Check for MZ signature.

        @type rd: L{ReadData}
        @param rd: A L{ReadData} object.

        @rtype: bool
        @return: True is the given L{ReadData} stream has the MZ signature. Otherwise, False.
        """
        rd.setOffset(0)
        sign = rd.read(2)
        if sign == "MZ":
            return True
        return False