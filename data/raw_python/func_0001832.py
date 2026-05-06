def get_sequence(self, c, i, depth):
        """
        Get the sequence.

        Get sequence between `{}`, such as: `{a,b}`, `{1..2[..inc]}`, etc.
        It will basically crawl to the end or find a valid series.
        """

        result = []
        release = self.set_expanding()
        has_comma = False  # Used to indicate validity of group (`{1..2}` are an exception).
        is_empty = True  # Tracks whether the current slot is empty `{slot,slot,slot}`.

        # Detect numerical and alphabetic series: `{1..2}` etc.
        i.rewind(1)
        item = self.get_range(i)
        i.advance(1)
        if item is not None:
            self.release_expanding(release)
            return (x for x in item)

        try:
            while c:
                # Bash has some special top level logic. if `}` follows `{` but hasn't matched
                # a group yet, keep going except when the first 2 bytes are `{}` which gets
                # completely ignored.
                keep_looking = depth == 1 and not has_comma  # and i.index not in self.skip_index
                if (c == '}' and (not keep_looking or i.index == 2)):
                    # If there is no comma, we know the sequence is bogus.
                    if is_empty:
                        result = (x for x in self.combine(result, ['']))
                    if not has_comma:
                        result = ('{' + literal + '}' for literal in result)
                    self.release_expanding(release)
                    return (x for x in result)

                elif c == ',':
                    # Must be the first element in the list.
                    has_comma = True
                    if is_empty:
                        result = (x for x in self.combine(result, ['']))
                    else:
                        is_empty = True

                else:
                    if c == '}':
                        # Top level: If we didn't find a comma, we haven't
                        # completed the top level group. Request more and
                        # append to what we already have for the first slot.
                        if not result:
                            result = (x for x in self.combine(result, [c]))
                        else:
                            result = self.squash(result, [c])
                        value = self.get_literals(next(i), i, depth)
                        if value is not None:
                            result = self.squash(result, value)
                            is_empty = False
                    else:
                        # Lower level: Try to find group, but give up if cannot acquire.
                        value = self.get_literals(c, i, depth)
                        if value is not None:
                            result = (x for x in self.combine(result, value))
                            is_empty = False

                c = next(i)
        except StopIteration:
            self.release_expanding(release)
            raise