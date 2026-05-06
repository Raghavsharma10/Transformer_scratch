def html_to_text(html, base_url='', bodywidth=CONFIG_DEFAULT):
    """
    Convert a HTML mesasge to plain text.
    """
    def _patched_handle_charref(c):
        self = h
        charref = self.charref(c)
        if self.code or self.pre:
            charref = cgi.escape(charref)
        self.o(charref, 1)

    def _patched_handle_entityref(c):
        self = h
        entityref = self.entityref(c)
        if self.code or self.pre:  # this expression was inversed.
            entityref = cgi.escape(entityref)
        self.o(entityref, 1)

    h = HTML2Text(baseurl=base_url, bodywidth=config.BODY_WIDTH if bodywidth is CONFIG_DEFAULT else bodywidth)
    h.handle_entityref = _patched_handle_entityref
    h.handle_charref = _patched_handle_charref
    return h.handle(html).rstrip()