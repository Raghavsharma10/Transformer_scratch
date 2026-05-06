def __write(self, thePath, theData):
        """
        Write data to a file.
        
        @type thePath: str
        @param thePath: The file path.
        
        @type theData: str
        @param theData: The data to write.
        """    
        fd = open(thePath, "wb")
        fd.write(theData)
        fd.close()