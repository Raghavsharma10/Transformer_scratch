def select(self):
        """First part of an SQL query."""
        # Try to match the asterisk, any or list of vars.
        if self.tokens.accept(grammar.select_any):
            return self.select_any()

        if self.tokens.accept(grammar.select_all):
            # The FROM after SELECT * is required.
            self.tokens.expect(grammar.select_from)
            return self.select_from()

        return self.select_what()