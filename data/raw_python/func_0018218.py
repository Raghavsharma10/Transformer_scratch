def process_escape(self, char):
        '''Handle the char after the escape char'''
        # Always only run once, switch back to the last processor.
        self.process_char = self.last_process_char
        if self.part == [] and char in self.whitespace:
            # Special case where \ is by itself and not at the EOL.
            self.parts.append(self.escape_char)
            return
        if char == self.eol_char:
            # Ignore a cr.
            return
        unescaped = self.escape_results.get(char, self.escape_char+char)
        self.part.append(unescaped)