def methods(self, *args, **kwds):
        """
        Request info of callable remote methods.

        Arguments for :meth:`call` except for `name` can be applied to
        this function too.

        """
        self.callmanager.methods(self, *args, **kwds)