def bundle_view(parser, token):
    """
    Returns an string version of a bundle view. This is done by
    calling the `get_string_from_view` method of the provided bundle.

    This tag expects that the request object as well as the
    the original url_params are available in the context.

    Requires two arguments bundle and the name of the view
    you want to render. In addition, this tag also accepts
    the 'as xxx' syntax.

    Example:

    {% bundle_url bundle main_list as html %}
    """

    bits = token.split_contents()
    if len(bits) < 3:
        raise TemplateSyntaxError("'%s' takes at least two arguments"
                                  " bundle and view_name" % bits[0])

    bundle = parser.compile_filter(bits[1])
    viewname = parser.compile_filter(bits[2])

    asvar = None
    bits = bits[2:]
    if len(bits) >= 2 and bits[-2] == 'as':
        asvar = bits[-1]
        bits = bits[:-2]

    return ViewNode(bundle, viewname, asvar)