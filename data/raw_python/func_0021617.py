def getWordAtOffset(self, offset):
        """
        Returns a C{WORD} from a given offset. 
        
        @type offset: int
        @param offset: The offset to get the C{WORD} from.
        
        @rtype: L{WORD}
        @return: The L{WORD} obtained at the given offset.
        """
        return datatypes.WORD.parse(utils.ReadData(self.getDataAtOffset(offset, 2)))