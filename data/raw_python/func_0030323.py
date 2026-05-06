def withdict(parser, token):
    """
    Take a complete context dict as extra layer.
    """
    bits = token.split_contents()
    if len(bits) != 2:
        raise TemplateSyntaxError("{% withdict %} expects one argument")

    nodelist = parser.parse(('endwithdict',))
    parser.delete_first_token()

    return WithDictNode(
        nodelist=nodelist,
        context_expr=parser.compile_filter(bits[1])
    )