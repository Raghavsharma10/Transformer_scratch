def unquote_string(self, string):
        """Return the unquoted value of a quoted string."""
        value = string[1:-1]

        forbidden_sequences = {ESCAPE_SUBS[STRING_QUOTES[string[0]]]}
        valid_sequences = set(ESCAPE_SEQUENCES) - forbidden_sequences

        for seq in ESCAPE_REGEX.findall(value):
            if seq not in valid_sequences:
                raise self.error(f'Invalid escape sequence "{seq}"')

        for seq, sub in ESCAPE_SEQUENCES.items():
            value = value.replace(seq, sub)

        return value