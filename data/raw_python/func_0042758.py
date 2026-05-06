def add_tokens_for_pass(self):
        """Add tokens for a pass to result"""
        # Make sure pass not added to group again
        self.groups.empty = False

        # Remove existing newline/indentation
        while self.result[-1][0] in (INDENT, NEWLINE):
            self.result.pop()

        # Add pass and indentation
        self.add_tokens(
            [ (NAME, 'pass')
            , (NEWLINE, '\n')
            , (INDENT, self.indent_type * self.current.scol)
            ]
        )