def match(self, f, *args):
        """Match grammar function 'f' against next token and set 'self.matched'.

        Arguments:
            f: A grammar function - see efilter.parsers.common.grammar. Must
                return TokenMatch or None.
            args: Passed to 'f', if any.

        Returns:
            Instance of efilter.parsers.common.grammar.TokenMatch or None.

        Comment:
            If a match is returned, it will also be stored in self.matched.
        """
        try:
            match = f(self.tokenizer, *args)
        except StopIteration:
            # The grammar function might have tried to access more tokens than
            # are available. That's not really an error, it just means it didn't
            # match.
            return

        if match is None:
            return

        if not isinstance(match, grammar.TokenMatch):
            raise TypeError("Invalid grammar function %r returned %r."
                            % (f, match))

        self.matched = match
        return match