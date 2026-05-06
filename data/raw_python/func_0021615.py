def getWordAtRva(self, rva):
        """
        Returns a C{WORD} from a given RVA. 
        
        @type rva: int
        @param rva: The RVA to get the C{WORD} from.
        
        @rtype: L{WORD}
        @return: The L{WORD} obtained at the given RVA.
        """
        return datatypes.WORD.parse(utils.ReadData(self.getDataAtRva(rva,  2)))