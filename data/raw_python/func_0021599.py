def readFile(self, pathToFile):
        """
        Returns data from a file.
        
        @type pathToFile: str
        @param pathToFile: Path to the file.
        
        @rtype: str
        @return: The data from file.
        """
        fd = open(pathToFile,  "rb")
        data = fd.read()
        fd.close()
        return data