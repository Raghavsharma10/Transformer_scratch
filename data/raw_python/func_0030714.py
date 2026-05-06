def recursiveRepr(self, stuff, thunk=repr):
        """
        Recursive repr().
        """
        ID = id(stuff)
        if ID in self.active:
            return '%s(...)' % (stuff.__class__.__name__,)
        else:
            try:
                self.active[ID] = stuff
                return thunk(stuff)
            finally:
                del self.active[ID]