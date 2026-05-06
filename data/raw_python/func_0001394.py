def msw(self):
        ''' Return a generator of tokens with more than one sense. '''
        return (t for t, c in self.tcmap().items() if len(c) > 1)