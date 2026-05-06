def _input_as_lines(self, data):
        """ Write a seq of lines to a temp file and return the filename string

            data: a sequence to be written to a file, each element of the
                sequence will compose a line in the file
           * Note: the result will be the filename as a FilePath object
            (which is a string subclass).

           * Note: '\n' will be stripped off the end of each sequence element
                before writing to a file in order to avoid multiple new lines
                accidentally be written to a file
        """
        filename = self._input_filename = \
            FilePath(self.getTmpFilename(self.TmpDir))
        filename = FilePath(filename)
        data_file = open(filename, 'w')
        data_to_file = '\n'.join([str(d).strip('\n') for d in data])
        data_file.write(data_to_file)
        data_file.close()
        return filename