def getQwordAtRva(self, rva):
        """
        Returns a C{QWORD} from a given RVA. 
        
        @type rva: int
        @param rva: The RVA to get the C{QWORD} from.
        
        @rtype: L{QWORD}
        @return: The L{QWORD} obtained at the given RVA.
        """
        return datatypes.QWORD.parse(utils.ReadData(self.getDataAtRva(rva,  8)))