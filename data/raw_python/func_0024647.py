def _char_diff(self, old, new, transition, fmt=lambda c: c):
        """
        Returns a char-based diff between `old` and `new` where each character
        is formatted by `fmt` and transitions between blocks are determined by `transition`.
        """

        differ = difflib.ndiff(old, new)

        # Type of difference.
        dtype = None

        # Buffer for current line.
        line = []
        while True:
            # Get next diff or None if we're at the end.
            d = next(differ, (None,))
            if d[0] != dtype:
                line += transition(dtype, d[0])
                dtype = d[0]

            if dtype is None:
                break

            if d[2] == "\n":
                if dtype != " ":
                    self._warn_chars.add((dtype, "\\n"))
                    # Show added/removed newlines.
                    line += [fmt(r"\n"), transition(dtype, " ")]

                # Don't yield a line if we are removing a newline
                if dtype != "-":
                    yield "".join(line)
                    line.clear()

                line.append(transition(" ", dtype))
            elif dtype != " " and d[2] == "\t":
                # Show added/removed tabs.
                line.append(fmt("\\t"))
                self._warn_chars.add((dtype, "\\t"))
            else:
                line.append(fmt(d[2]))

        # Flush buffer before quitting.
        last = "".join(line)
        # Only print last line if it contains non-ANSI characters.
        if re.sub(r"\x1b[^m]*m", "", last):
            yield last