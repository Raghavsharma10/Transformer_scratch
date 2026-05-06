def might_prefer(self, **items):
        '''Items to take precedence if their values are not None (never saved)'''
        self._overrides = dict((k, v) for (k, v) in items.items() if v is not None)