def html_to_xhtml(html_unicode_string):
    """
    Converts html to xhtml

    Args:
        html_unicode_string: A (possible unicode) string representing HTML.

    Returns:
        A (possibly unicode) string representing XHTML.

    Raises:
        TypeError: Raised if input_string isn't a unicode string or string.
    """
    try:
        assert isinstance(html_unicode_string, basestring)
    except AssertionError:
        raise TypeError
    root = BeautifulSoup(html_unicode_string, 'html.parser')
    # Confirm root node is html
    try:
        assert root.html is not None
    except AssertionError:
        raise ValueError(''.join(['html_unicode_string cannot be a fragment.',
                         'string is the following: %s', unicode(root)]))
    # Add xmlns attribute to html node
    root.html['xmlns'] = 'http://www.w3.org/1999/xhtml'
    unicode_string = unicode(root.prettify(encoding='utf-8', formatter='html'), encoding='utf-8')
    # Close singleton tag_dictionary
    for tag in constants.SINGLETON_TAG_LIST:
        unicode_string = unicode_string.replace(
                '<' + tag + '/>',
                '<' + tag + ' />')
    return unicode_string