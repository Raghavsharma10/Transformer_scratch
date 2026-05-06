def search(self, text, lookup=None):
        '''Returns a new :class:`Query` for :attr:`Manager.model` with
a full text search value.'''
        return self.query().search(text, lookup=lookup)