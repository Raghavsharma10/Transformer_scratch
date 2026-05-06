def process_delimiter(self, char):
        '''Process chars while not in a part'''
        if char in self.whitespace:
            return
        if char in self.quote_chars:
            # Store the quote type (' or ") and switch to quote processing.
            self.inquote = char
            self.process_char = self.process_quote
            return
        if char == self.eol_char:
            self.complete = True
            return
        # Switch to processing a part.
        self.process_char = self.process_part
        self.process_char(char)