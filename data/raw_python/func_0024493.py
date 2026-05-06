def write(self, data):
        """
        write single molecule or reaction into file
        """
        self._file.write('<cml>')
        self.__write(data)
        self.write = self.__write