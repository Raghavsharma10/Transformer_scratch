def _add_punctuation_spacing(self, input):
        '''
        Adds additional spacing to punctuation characters. For example,
        this puts an extra space after a fullwidth full stop.
        '''
        for replacement in punct_spacing:
            input = re.sub(replacement[0], replacement[1], input)

        return input