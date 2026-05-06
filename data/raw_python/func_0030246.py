def options(self):
        '''
        Yields `(raw_value, label)` pairs for all acceptable choices.
        '''
        conv = self.conv
        for python_value, label in self.choices:
            yield conv.from_python(python_value), label