def _input_as_lines(self, data):
        """Writes data to tempfile and sets -i parameter

        data -- list of lines, ready to be written to file
        """
        if data:
            self.Parameters['-i']\
                    .on(super(CD_HIT,self)._input_as_lines(data))
        return ''