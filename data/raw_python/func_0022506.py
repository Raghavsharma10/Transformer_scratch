def expect(self, f, *args):
        """Like 'accept' but throws a parse error if 'f' doesn't match."""
        match = self.accept(f, *args)
        if match:
            return match

        try:
            func_name = f.func_name
        except AttributeError:
            func_name = "<unnamed grammar function>"

        start, end = self.current_position()
        raise errors.EfilterParseError(
            query=self.tokenizer.source, start=start, end=end,
            message="Was expecting %s here." % (func_name))