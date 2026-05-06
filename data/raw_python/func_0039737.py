def term(term, xpath=None, escape=True):
    """ Escapes <, > and & characters in the given term for inclusion into XML (like the search query).
        Also wrap the term in XML tags if xpath is specified.
        Note that this function doesn't escape the @, $, " and other symbols that are meaningful in a search query.

        Args:
            term -- The term text to be escaped (e.g. a search query term).

        Keyword args:
            xpath -- An optional xpath, to be specified if the term is to wraped in tags.
            escape -- An optional parameter - whether to escape the term's XML characters. Default is True.

        Returns:
            Properly escaped xml string for queries.

    >>> term('lorem<4')
    'lorem&lt;4'

    >>> term('3 < bar < 5 $$ True', 'document/foo', False)
    '<document><foo>3 < bar < 5 $$ True</foo></document>'

    >>> term('3 < bar < 5 $$ True', 'document/foo')
    '<document><foo>3 &lt; bar &lt; 5 $$ True</foo></document>'
    """
    prefix = []
    postfix = []
    if xpath:
        tags = xpath.split('/')
        for tag in tags:
            if tag:
                prefix.append('<{0}>'.format(tag))
                postfix.insert(0, '</{0}>'.format(tag))
    if escape:
        term = cgi.escape(term)
    return ''.join(prefix + [term] + postfix)