def get_literals(self, c, i, depth):
        """
        Get a string literal.

        Gather all the literal chars up to opening curly or closing brace.
        Also gather chars between braces and commas within a group (is_expanding).
        """

        result = ['']
        is_dollar = False

        try:
            while c:
                ignore_brace = is_dollar
                is_dollar = False

                if c == '$':
                    is_dollar = True

                elif c == '\\':
                    c = [self.get_escape(c, i)]

                elif not ignore_brace and c == '{':
                    # Try and get the group
                    index = i.index
                    try:
                        seq = self.get_sequence(next(i), i, depth + 1)
                        if seq:
                            c = seq
                    except StopIteration:
                        # Searched to end of string
                        # and still didn't find it.
                        i.rewind(i.index - index)

                elif self.is_expanding() and c in (',', '}'):
                    # We are Expanding within a group and found a group delimiter
                    # Return what we gathered before the group delimiters.
                    i.rewind(1)
                    return (x for x in result)

                # Squash the current set of literals.
                result = self.squash(result, [c] if isinstance(c, str) else c)

                c = next(i)
        except StopIteration:
            if self.is_expanding():
                return None

        return (x for x in result)