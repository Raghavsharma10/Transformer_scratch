def to_romaji(self, input):
        '''
        Converts kana input to rōmaji and returns the result.
        '''
        input = self._preprocess_input(input)

        # Preprocess the input, making string replacements where needed.
        chars = list(input)
        chars = self._preprocess_chars(chars)

        chars.append(END_CHAR)
        for char in chars:
            if char in di_a:
                self._set_digraph_a(char)
                continue

            if char in di_b:
                self._set_digraph_b(char)
                continue

            if char in cvs:
                self._set_char(char, CV)
                continue

            if char in vowels:
                self._set_vowel(char)
                continue

            if char in xvowels:
                self._set_xvowel(char)
                continue

            if char in geminates:
                self._inc_geminate()
                continue

            if char == lvmarker:
                self._inc_lvmarker()
                continue

            if char == WORD_BORDER:
                # When stumbling upon a word border, e.g. in ぬれ|えん,
                # the current word has finished, meaning the character
                # should be flushed.
                self._flush_char()
                continue

            if char == END_CHAR:
                self._promote_solitary_xvowel()
                self._flush_char()
                continue

            # If we're still here, that means we've stumbled upon a character
            # the machine can't deal with.
            if self.unknown_strategy == UNKNOWN_DISCARD:
                continue

            if self.unknown_strategy == UNKNOWN_RAISE:
                raise UnexpectedCharacterError

            if self.unknown_strategy == UNKNOWN_INCLUDE:
                # The default strategy.
                self._add_unknown_char(char)

        return self._flush_stack()