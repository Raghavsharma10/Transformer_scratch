def python_data(self):
        '''Representation of aggregate value as dictionary.'''
        try:
            value = self.clean_value
        except LookupError:
            # XXX is this necessary?
            value = self.get_initial()
        return self.from_python(value)