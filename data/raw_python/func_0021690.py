def get_prep_value(self, value):
        '''Prepare value for database storage.'''
        if isinstance(value, Seconds):
            return value.seconds
        elif value:
            return self.parse_seconds(value).seconds
        else:
            return None