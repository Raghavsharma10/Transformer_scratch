def getQwordAtOffset(self, offset):
        """
        Returns a C{QWORD} from a given offset. 
        
        @type offset: int
        @param offset: The offset to get the C{QWORD} from.
        
        @rtype: L{QWORD}
        @return: The L{QWORD} obtained at the given offset.
        """
        return datatypes.QWORD.parse(utils.ReadData(self.getDataAtOffset(offset,  8)))