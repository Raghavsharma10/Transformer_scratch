def render_rootURL(self, ctx, data):
        """
        Add the WebSite's root URL as a child of the given tag.
        """
        return ctx.tag[
            ixmantissa.ISiteURLGenerator(self.store).rootURL(IRequest(ctx))]