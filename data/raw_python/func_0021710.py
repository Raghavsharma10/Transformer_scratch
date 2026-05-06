def escape_string(self, string):
        """Return the escaped literal representation of an nbt string."""
        if self.quote:
            quote = self.quote
        else:
            found = QUOTE_REGEX.search(string)
            quote = STRING_QUOTES[found.group()] if found else next(iter(STRING_QUOTES))

        for match, seq in ESCAPE_SUBS.items():
            if match == quote or match not in STRING_QUOTES:
                string = string.replace(match, seq)

        return f'{quote}{string}{quote}'