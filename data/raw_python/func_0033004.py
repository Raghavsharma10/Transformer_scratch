def render_footer(self, ctx, data):
        """
        Render any required static content in the footer, from the C{staticContent}
        attribute of this page.
        """
        if self.staticContent is None:
            return ctx.tag

        header = self.staticContent.getFooter()
        if header is not None:
            return ctx.tag[header]
        else:
            return ctx.tag