def plugitInclude(parser, token):
    """
        Load and render a template, using the same context of a specific action.

        Example: {% plugitInclude "/menuBar" %}
    """
    bits = token.split_contents()

    if len(bits) != 2:
        raise TemplateSyntaxError("'plugitInclude' tag takes one argument: the tempalte's action to use")

    action = parser.compile_filter(bits[1])

    return PlugItIncludeNode(action)