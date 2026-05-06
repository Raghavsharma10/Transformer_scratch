def to_python(self, value):
        '''Handle data from serialization and form clean() methods.'''
        if isinstance(value, Seconds):
            return value
        if value in self.empty_values:
            return None
        return self.parse_seconds(value)