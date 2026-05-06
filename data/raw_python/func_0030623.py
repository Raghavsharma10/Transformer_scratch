def qs_get(self, key, default=None):
        '''Get a value from QuerySet MultiDict'''
        return self.query.get(key, default=default)