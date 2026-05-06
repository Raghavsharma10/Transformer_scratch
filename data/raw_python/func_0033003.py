def render_header(self, ctx, data):
        """
        Render any required static content in the header, from the C{staticContent}
        attribute of this page.
        """
        if self.staticContent is None:
            return ctx.tag

        header = self.staticContent.getHeader()
        if header is not None:
            return ctx.tag[header]
        else:
            return ctx.tag