def add_phase(self):
        """Context manager for when adding all the tokens"""
        # add stuff
        yield self

        # Make sure we output eveything
        self.finish_hanging()

        # Remove trailing indents and dedents
        while len(self.result) > 1 and self.result[-2][0] in (INDENT, ERRORTOKEN, NEWLINE):
            self.result.pop(-2)