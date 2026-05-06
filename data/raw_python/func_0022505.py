def reject(self, f, *args):
        """Like 'match', but throw a parse error if 'f' matches.

        This is useful when a parser wants to be strict about specific things
        being prohibited. For example, DottySQL bans the use of SQL keywords as
        variable names.
        """
        match = self.match(f, *args)
        if match:
            token = self.peek(0)
            raise errors.EfilterParseError(
                query=self.tokenizer.source, token=token,
                message="Was not expecting a %s here." % token.name)