def write(self, filename = ""):
        """
        Writes data from L{PE} object to a file.
        
        @rtype: str
        @return: The L{PE} stream data.

        @raise IOError: If the file could not be opened for write operations.
        """
        file_data = str(self)
        if filename:
            try:
                self.__write(filename, file_data)
            except IOError:
                raise IOError("File could not be opened for write operations.")
        else:
            return file_data