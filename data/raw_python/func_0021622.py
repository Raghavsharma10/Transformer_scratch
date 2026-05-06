def readStringAtRva(self, rva):
        """
        Returns a L{String} object from a given RVA. 
        
        @type rva: int
        @param rva: The RVA to get the string from.
        
        @rtype: L{String}
        @return: A new L{String} object from the given RVA.
        """
        d = self.getDataAtRva(rva,  1)
        resultStr = datatypes.String("")
        while d != "\x00":
            resultStr.value += d
            rva += 1
            d = self.getDataAtRva(rva, 1)
        return resultStr