def z__update(self):
        """Triple quoted baseline representation.

        Return string with multiple triple quoted baseline strings when
        baseline had been compared multiple times against varying strings.

        :returns: source file baseline replacement text
        :rtype: str

        """
        updates = []

        for text in self._updates:

            if self._AVOID_RAW_FORM:
                text_repr = multiline_repr(text)
                raw_char = ''
            else:
                text_repr = multiline_repr(text, RAW_MULTILINE_CHARS)

                if len(text_repr) == len(text):
                    raw_char = 'r' if '\\' in text_repr else ''
                else:
                    # must have special characters that required added backslash
                    # escaping, use normal representation to get backslashes right
                    text_repr = multiline_repr(text)
                    raw_char = ''

            # use triple double quote, except use triple single quote when
            # triple double quote is present to avoid syntax errors
            quotes = '"""'
            if quotes in text:
                quotes = "'''"

            # Wrap with blank lines when multi-line or when text ends with
            # characters that would otherwise result in a syntax error in
            # the formatted representation.
            multiline = self._indent or ('\n' in text)
            if multiline or text.endswith('\\') or text.endswith(quotes[0]):
                update = raw_char + quotes + '\n' + text_repr + '\n' + quotes
            else:
                update = raw_char + quotes + text_repr + quotes

            updates.append(update)

        # sort updates so Python hash seed has no impact on regression test
        update = '\n'.join(sorted(updates))

        indent = ' ' * self._indent

        lines = ((indent + line) if line else '' for line in update.split('\n'))

        return '\n'.join(lines).lstrip()