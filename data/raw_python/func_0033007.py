def render_head(self, ctx, data):
        """
        This renderer calculates content for the <head> tag by concatenating the
        values from L{getHeadContent} and the overridden L{head} method.
        """
        req = inevow.IRequest(ctx)
        more = getattr(self.fragment, 'head', None)
        if more is not None:
            fragmentHead = more()
        else:
            fragmentHead = None
        return ctx.tag[filter(None, list(self.getHeadContent(req)) +
                              [fragmentHead])]