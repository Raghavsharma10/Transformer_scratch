def getDwordAtRva(self, rva):
        """
        Returns a C{DWORD} from a given RVA. 
        
        @type rva: int
        @param rva: The RVA to get the C{DWORD} from.
        
        @rtype: L{DWORD}
        @return: The L{DWORD} obtained at the given RVA.
        """
        return datatypes.DWORD.parse(utils.ReadData(self.getDataAtRva(rva,  4)))