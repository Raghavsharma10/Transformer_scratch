def locateChild(self, ctx, segments):
        """
        Attempt to locate the child via the '.fragment' attribute, then fall
        back to normal locateChild behavior.
        """
        if self.fragment is not None:
            # There are still a bunch of bogus subclasses of this class, which
            # are used in a variety of distasteful ways.  'fragment' *should*
            # always be set to something that isn't None, but there's no way to
            # make sure that it will be for the moment.  Every effort should be
            # made to reduce public use of subclasses of this class (instead
            # preferring to wrap content objects with
            # IWebViewer.wrapModel()), so that the above check can be
            # removed. -glyph
            lc = getattr(self.fragment, 'locateChild', None)
            if lc is not None:
                x = lc(ctx, segments)
                if x is not NotFound:
                    return x
        return super(MantissaViewHelper, self).locateChild(ctx, segments)