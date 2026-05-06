def _input_as_multiline_string(self, data):
        """Writes data to tempfile and sets -infile parameter

        data -- list of lines
        """
        if data:
            self.Parameters['-infile']\
                .on(super(Clustalw,self)._input_as_multiline_string(data))
        return ''