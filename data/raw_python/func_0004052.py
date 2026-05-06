def parse_html(html):
    """ Create an lxml.html.HtmlElement from a string with html.
    XXX: mostly copy-pasted from parsel.selector.create_root_node
    """
    body = html.strip().replace('\x00', '').encode('utf8') or b'<html/>'
    parser = lxml.html.HTMLParser(recover=True, encoding='utf8')
    root = lxml.etree.fromstring(body, parser=parser)
    if root is None:
        root = lxml.etree.fromstring(b'<html/>', parser=parser)
    return root