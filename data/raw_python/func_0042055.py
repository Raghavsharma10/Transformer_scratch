def bundle_url(parser, token):
    """
    Returns an a url for given a bundle and a view name.
    This is done by calling the `get_view_url` method
    of the provided bundle.

    This tag expects that the request object as well as the
    the original url_params are available in the context.

    Requires two arguments bundle and the name of the view
    you want to render. In addition, this tag also accepts
    the 'as xxx' syntax.

    By default this tag will follow references to
    parent bundles. To stop this from happening pass
    `follow_parent=False`. Any other keyword arguments
    will be used as url keyword arguments.

    If no match is found a blank string will be returned.

    Example:

    {% bundle_url bundle "edit" obj=obj as html %}
    """

    bits = token.split_contents()
    if len(bits) < 3:
        raise TemplateSyntaxError("'%s' takes at least two arguments"
                                  " bundle and view_name" % bits[0])

    bundle = parser.compile_filter(bits[1])
    viewname = parser.compile_filter(bits[2])

    kwargs = {}
    asvar = None
    bits = bits[2:]
    if len(bits) >= 2 and bits[-2] == 'as':
        asvar = bits[-1]
        bits = bits[:-2]

    if len(bits):
        for bit in bits:
            match = kwarg_re.match(bit)
            if not match:
                raise TemplateSyntaxError("Malformed arguments to url tag")
            name, value = match.groups()
            if name:
                kwargs[name] = parser.compile_filter(value)

    return URLNode(bundle, viewname, kwargs, asvar)