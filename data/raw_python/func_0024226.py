def write(self, data):
        """
        write single molecule into file
        """
        m = self._convert_structure(data)
        self._file.write(self._format_mol(*m))
        self._file.write('M  END\n')

        for k, v in data.meta.items():
            self._file.write(f'>  <{k}>\n{v}\n')
        self._file.write('$$$$\n')