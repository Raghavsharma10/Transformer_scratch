def make_object(self, state=None, backend=None):
        '''Create a new instance of :attr:`model` from a *state* tuple.'''
        model = self.model
        obj = model.__new__(model)
        self.load_state(obj, state, backend)
        return obj