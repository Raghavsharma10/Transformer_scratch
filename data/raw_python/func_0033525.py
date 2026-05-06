def _input_as_multiline_string(self, data):
        """Writes data to tempfile and sets -i parameter

        data -- list of lines
        """
        if data:
            self.Parameters['-i']\
                    .on(super(CD_HIT,self)._input_as_multiline_string(data))
        return ''