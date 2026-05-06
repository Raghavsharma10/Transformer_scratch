def replace_baseline_repr(self, linenum, update):
        """Replace individual baseline representation.

        :param int linenum: location of baseline representation
        :param str update: new baseline representation text (with delimiters)

        """
        # use property to access lines to read them from file if necessary
        lines = self.lines

        count = 0
        delimiter = None
        for index in range(linenum - 1, -1, -1):
            line = lines[index]
            if delimiter is None:
                single_quote_index = line.rfind("'''")
                double_quote_index = line.rfind('"""')
                if double_quote_index >= 0:
                    if double_quote_index > single_quote_index:
                        delimiter = '"""'
                    else:
                        delimiter = "'''"
                elif single_quote_index >= 0:
                    delimiter = "'''"
                else:
                    continue
            count += lines[index].count(delimiter)
            if count >= 2:
                linenum = index
                break
        else:
            docstr_not_found = (
                '{}:{}: could not find baseline docstring'
                ''.format(self.showpath(self.path), linenum))
            raise RuntimeError(docstr_not_found)

        old_content = '\n'.join(lines[linenum:])

        match = self.REGEX.match(old_content)

        if match is None:
            docstr_not_found = (
                '{}:{}: could not find docstring'.format(self.path, linenum))
            raise RuntimeError(docstr_not_found)

        new_content = match.group('prefix') + update + match.group('suffix')

        lines[linenum:] = new_content.split('\n')