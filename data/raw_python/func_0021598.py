def hasPESignature(self, rd):
        """
        Check for PE signature.

        @type rd: L{ReadData}
        @param rd: A L{ReadData} object.

        @rtype: bool
        @return: True is the given L{ReadData} stream has the PE signature. Otherwise, False.
        """
        rd.setOffset(0)
        e_lfanew_offset = unpack("<L",  rd.readAt(0x3c, 4))[0]
        sign = rd.readAt(e_lfanew_offset, 2)
        if sign == "PE":
            return True
        return False