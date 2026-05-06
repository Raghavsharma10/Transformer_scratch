def process_quote(self, char):
        '''Process character while in a quote'''
        if char == self.inquote:
            # Quote is finished, switch to part processing.
            self.process_char = self.process_part
            return
        try:
            self.part.append(char)
        except:
            self.part = [ char ]