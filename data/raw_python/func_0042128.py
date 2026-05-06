def _preprocess_chars(self, chars):
        '''
        Performs string preprocessing before the main conversion algorithm
        is used. Simple string replacements (for example, fullwidth rōmaji
        to regular rōmaji) are performed at this point.
        '''
        chars = self._normalize_dakuten(chars)
        chars = self._process_repeaters(chars)
        chars = self._perform_replacements(chars)

        return chars