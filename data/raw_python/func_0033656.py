def _input_as_multiline_string(self, data):
        """Writes data to tempfile and sets -infile parameter

        data -- list of lines
        """
        if data:
            self.Parameters['--in']\
                .on(super(Clearcut,self)._input_as_multiline_string(data))
        return ''