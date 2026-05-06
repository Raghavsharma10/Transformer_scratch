def _preprocess_input(self, input):
        '''
        Preprocesses the input before it's split into a list.
        '''
        if not re.search(preprocess_chars, input):
            # No characters that we need to preprocess, so continue without.
            return input

        input = self._add_punctuation_spacing(input)

        return input