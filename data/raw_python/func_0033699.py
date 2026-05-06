def _input_as_lines(self, data):
        """Write sequence of lines to temp file, return filename

        data: a sequence to be written to a file, each element of the
            sequence will compose a line in the file

        * Note: '\n' will be stripped off the end of each sequence
            element before writing to a file in order to avoid
            multiple new lines accidentally be written to a file
        """
        self._input_filename = self.getTmpFilename(
            self.WorkingDir, suffix='.fasta')
        with open(self._input_filename, 'w') as f:
            # Use lazy iteration instead of list comprehension to
            # prevent reading entire file into memory
            for line in data:
                f.write(str(line).strip('\n'))
                f.write('\n')
        return self._input_filename