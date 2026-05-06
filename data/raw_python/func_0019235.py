def add(self, indent, line):
        """Appends the given text line with prefixed spaces in accordance with
        the given number of indentation levels.
        """
        if isinstance(line, str):
            list.append(self, indent*4*' ' + line)
        else:
            for subline in line:
                list.append(self, indent*4*' ' + subline)