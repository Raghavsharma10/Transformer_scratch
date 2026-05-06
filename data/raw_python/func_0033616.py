def _input_as_lines(self,data):
        """Writes data to tempfile and sets -infile parameter

        data -- list of lines, ready to be written to file
        """
        if data:
            self.Parameters['-infile']\
                .on(super(Clustalw,self)._input_as_lines(data))
        return ''