def remove_comments(xml):
    """
    Remove comments, as they can break the xml parser.

    See html5lib issue #122 ( http://code.google.com/p/html5lib/issues/detail?id=122 ).

    >>> remove_comments('<!-- -->')
    ''
    >>> remove_comments('<!--\\n-->')
    ''
    >>> remove_comments('<p>stuff<!-- \\n -->stuff</p>')
    '<p>stuffstuff</p>'
    """
    regex = re.compile(r'<!--.*?-->', re.DOTALL)
    return regex.sub('', xml)