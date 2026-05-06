def convert_dedent(self):
        """Convert a dedent into an indent"""
        # Dedent means go back to last indentation
        if self.indent_amounts:
            self.indent_amounts.pop()

        # Change the token
        tokenum = INDENT

        # Get last indent amount
        last_indent = 0
        if self.indent_amounts:
            last_indent = self.indent_amounts[-1]

        # Make sure we don't have multiple indents in a row
        while self.result[-1][0] == INDENT:
            self.result.pop()

        value = self.indent_type * last_indent
        return tokenum, value