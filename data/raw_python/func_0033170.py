def _input_as_multiline_string(self, data):
        """Write a multiline string to a temp file and return the filename.

            data: a multiline string to be written to a file.

           * Note: the result will be the filename as a FilePath object
            (which is a string subclass).

        """
        filename = self._input_filename = \
            FilePath(self.getTmpFilename(self.TmpDir))
        data_file = open(filename, 'w')
        data_file.write(data)
        data_file.close()
        return filename