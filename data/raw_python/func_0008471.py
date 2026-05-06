def _lazy(self, method, *args):
        """ If the list is empty, calls lazylist.load().
            Replaces lazylist.method() with list.method() and calls it.
        """
        if list.__len__(self) == 0:
            self.load()
            setattr(self, method, types.MethodType(getattr(list, method), self))
        return getattr(list, method)(self, *args)