def expand(self, string):
        """Expand."""

        self.expanding = False
        empties = []
        found_literal = False
        if string:
            i = iter(StringIter(string))
            for x in self.get_literals(next(i), i, 0):
                # We don't want to return trailing empty strings.
                # Store empty strings and output only when followed by a literal.
                if not x:
                    empties.append(x)
                    continue
                found_literal = True
                while empties:
                    yield empties.pop(0)
                yield x
        empties = []

        # We found no literals so return an empty string
        if not found_literal:
            yield ""