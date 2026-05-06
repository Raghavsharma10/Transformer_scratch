def cleaned_selector(html):
    """ Clean parsel.selector.
    """
    import parsel
    try:
        tree = _cleaned_html_tree(html)
        sel = parsel.Selector(root=tree, type='html')
    except (lxml.etree.XMLSyntaxError,
            lxml.etree.ParseError,
            lxml.etree.ParserError,
            UnicodeEncodeError):
        # likely plain text
        sel = parsel.Selector(html)
    return sel