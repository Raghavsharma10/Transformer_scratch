def inline_css(self, html):
        """Inlines CSS defined in external style sheets.
        """
        premailer = Premailer(html)
        inlined_html = premailer.transform(pretty_print=True)
        return inlined_html