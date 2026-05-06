def _postprocess_output(self, output):
        '''
        Performs the last modifications before the output is returned.
        '''
        # Replace long vowels with circumflex characters.
        if self.vowel_style == CIRCUMFLEX_STYLE:
            try:
                output = output.translate(vowels_to_circumflexes)
            except TypeError:
                # Python 2 will error out here if there are no
                # macron characters in the string to begin with.
                pass

        # Output the desired case.
        if self.uppercase:
            output = output.upper()

        return output