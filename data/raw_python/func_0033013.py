def locateChild(self, ctx, segments):
        """
        Return a clone of this page that remembers its segments, so that URLs like
        /login/private/stuff will redirect the user to /private/stuff after
        login has completed.
        """
        arguments = IRequest(ctx).args
        return self.__class__(
            self.store, segments, arguments), ()