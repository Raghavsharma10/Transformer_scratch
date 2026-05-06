def _input_as_parameter(self, data):
        """ Set the input path and log path based on data (a fasta filepath)
        """
        self.Parameters['-i'].on(data)
        # access data through self.Parameters so we know it's been cast
        # to a FilePath
        input_filepath = self.Parameters['-i'].Value
        input_file_dir, input_filename = split(input_filepath)
        input_file_base, input_file_ext = splitext(input_filename)
        # FIXME: the following all other options
        # formatdb ignores the working directory if not name is passed.
        self.Parameters['-l'].on(FilePath('%s.log') % input_filename)
        self.Parameters['-n'].on(FilePath(input_filename))
        return ''