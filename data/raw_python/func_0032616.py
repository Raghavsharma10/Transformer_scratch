def render_rootURL(self, ctx, data):
        """
        Add the WebSite's root URL as a child of the given tag.

        The root URL is the location of the resource beneath which all standard
        Mantissa resources (such as the private application and static content)
        is available.  This can be important if a page is to be served at a
        location which is different from the root URL in order to make links in
        static XHTML templates resolve correctly (for example, by adding this
        value as the href of a <base> tag).
        """
        site = ISiteURLGenerator(self._siteStore())
        return ctx.tag[site.rootURL(IRequest(ctx))]