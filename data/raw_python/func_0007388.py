def tag(tagname, content='', attrs=None):
    """ Helper for programmatically building HTML tags.

    Note that this barely does any escaping, and will happily spit out
    dangerous user input if used as such.

    :param tagname:
        Tag name of the DOM element we want to return.

    :param content:
        Optional content of the DOM element. If `None`, then the element is
        self-closed. By default, the content is an empty string. Supports
        iterables like generators.

    :param attrs:
        Optional dictionary-like collection of attributes for the DOM element.

    Example::

        >>> tag('div', content='Hello, world.')
        u'<div>Hello, world.</div>'
        >>> tag('script', attrs={'src': '/static/js/core.js'})
        u'<script src="/static/js/core.js"></script>'
        >>> tag('script', attrs=[('src', '/static/js/core.js'), ('type', 'text/javascript')])
        u'<script src="/static/js/core.js" type="text/javascript"></script>'
        >>> tag('meta', content=None, attrs=dict(content='"quotedquotes"'))
        u'<meta content="\\\\"quotedquotes\\\\"" />'
        >>> tag('ul', (tag('li', str(i)) for i in xrange(3)))
        u'<ul><li>0</li><li>1</li><li>2</li></ul>'
    """
    attrs_str = attrs and ' '.join(_generate_dom_attrs(attrs))
    open_tag = tagname
    if attrs_str:
        open_tag += ' ' + attrs_str

    if content is None:
        return literal('<%s />' % open_tag)

    content = ''.join(iterate(content, unless=(basestring, literal)))
    return literal('<%s>%s</%s>' % (open_tag, content, tagname))