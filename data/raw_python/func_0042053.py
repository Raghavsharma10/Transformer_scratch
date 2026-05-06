def render_view(parser, token):
    """
    Return an string version of a View with as_string method.
    First argument is the name of the view. Any other arguments
    should be keyword arguments and will be passed to the view.

    Example:

    {% render_view viewname var1=xx var2=yy %}
    """
    bits = token.split_contents()

    n = len(bits)
    if n < 2:
        raise TemplateSyntaxError("'%s' takes at least one view as argument")

    viewname = bits[1]

    kwargs = {}
    if n > 2:
        for bit in bits[2:]:
            match = kwarg_re.match(bit)
            if not match:
                raise TemplateSyntaxError("Malformed arguments to render_view tag")
            name, value = match.groups()
            if name:
                kwargs[name] = parser.compile_filter(value)

    return StringNode(viewname, kwargs)