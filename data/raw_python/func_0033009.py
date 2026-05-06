def locateChild(self, ctx, segments):
        """
        Look up children in the normal manner, but then customize them for the
        authenticated user if they support the L{ICustomizable} interface.  If
        the user is attempting to access a private URL, redirect them.
        """
        result = self._getAppStoreResource(ctx, segments[0])
        if result is not None:
            child, segments = result, segments[1:]
            return child, segments

        if segments[0] == '':
            result = self.child_(ctx)
            if result is not None:
                child, segments = result, segments[1:]
                return child, segments

        # If the user is trying to access /private/*, then his session has
        # expired or he is otherwise not logged in. Redirect him to /login,
        # preserving the URL segments, rather than giving him an obscure 404.
        if segments[0] == 'private':
            u = URL.fromContext(ctx).click('/').child('login')
            for seg in segments:
                u = u.child(seg)
            return u, ()

        return rend.NotFound