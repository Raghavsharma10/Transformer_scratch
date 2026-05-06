def process(self, line):
        '''Step through the line and process each character'''
        self.raw = self.raw + line
        try:
            if not line[-1] == self.eol_char:
                # Should always be here, but add it just in case.
                line = line + self.eol_char
        except IndexError:
            # Thrown if line == ''
            line = self.eol_char
                
        for char in line:
            if char == self.escape_char:
                # Always handle escaped characters.
                self.last_process_char = self.process_char
                self.process_char = self.process_escape
                continue
            self.process_char(char)
        if not self.complete:
            # Ask for more.
            self.process( self.handler.readline(prompt=self.handler.CONTINUE_PROMPT) )