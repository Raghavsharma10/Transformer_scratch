def _perform_replacements(self, chars):
        '''
        Performs simple key/value string replacements that require no logic.
        This is used to convert the fullwidth rōmaji, several ligatures,
        and the punctuation characters.
        '''
        for n in range(len(chars)):
            char = chars[n]
            if char in repl:
                chars[n] = repl[char]

        # Some replacements might result in multi-character strings
        # being inserted into the list. Ensure we still have a list
        # of single characters for iteration.
        return list(''.join(chars))