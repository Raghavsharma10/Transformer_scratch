def getDwordAtOffset(self, offset):
        """
        Returns a C{DWORD} from a given offset. 
        
        @type offset: int
        @param offset: The offset to get the C{DWORD} from.
        
        @rtype: L{DWORD}
        @return: The L{DWORD} obtained at the given offset.
        """
        return datatypes.DWORD.parse(utils.ReadData(self.getDataAtOffset(offset,  4)))