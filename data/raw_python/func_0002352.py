def clean_html(input, sanitize=False):
    """
    Takes an HTML fragment and processes it using html5lib to ensure that the HTML is well-formed.

    :param sanitize: Remove unwanted HTML tags and attributes.

    >>> clean_html("<p>Foo<b>bar</b></p>")
    u'<p>Foo<b>bar</b></p>'
    >>> clean_html("<p>Foo<b>bar</b><i>Ooops!</p>")
    u'<p>Foo<b>bar</b><i>Ooops!</i></p>'
    >>> clean_html('<p>Foo<b>bar</b>& oops<a href="#foo&bar">This is a <>link</a></p>')
    u'<p>Foo<b>bar</b>&amp; oops<a href=#foo&amp;bar>This is a &lt;&gt;link</a></p>'
    """
    parser_kwargs = {}
    serializer_kwargs = {}
    if sanitize:
        if HTMLSanitizer is None:
            # new syntax as of 0.99999999/1.0b9 (Released on July 14, 2016)
            serializer_kwargs['sanitize'] = True
        else:
            parser_kwargs['tokenizer'] = HTMLSanitizer

    p = HTMLParser(tree=treebuilders.getTreeBuilder("dom"), **parser_kwargs)
    dom_tree = p.parseFragment(input)
    walker = treewalkers.getTreeWalker("dom")
    stream = walker(dom_tree)

    s = HTMLSerializer(omit_optional_tags=False, **serializer_kwargs)
    return "".join(s.serialize(stream))