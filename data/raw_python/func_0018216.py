def process_part(self, char):
        '''Process chars while in a part'''
        if char in self.whitespace or char == self.eol_char:
            # End of the part.
            self.parts.append( ''.join(self.part) )
            self.part = []
            # Switch back to processing a delimiter.
            self.process_char = self.process_delimiter
            if char == self.eol_char:
                self.complete = True
            return
        if char in self.quote_chars:
            # Store the quote type (' or ") and switch to quote processing.
            self.inquote = char
            self.process_char = self.process_quote
            return
        self.part.append(char)