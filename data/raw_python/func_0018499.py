def staticfile_node(parser, token, optimize_if_possible=False):
    """For example:
         {% staticfile "/js/foo.js" %}
         or
         {% staticfile "/js/foo.js" as variable_name %}
    Or for multiples:
         {% staticfile "/foo.js; /bar.js" %}
         or
         {% staticfile "/foo.js; /bar.js" as variable_name %}
    """
    args = token.split_contents()
    tag = args[0]

    if len(args) == 4 and args[-2] == 'as':
        context_name = args[-1]
        args = args[:-2]
    else:
        context_name = None

    filename = parser.compile_filter(args[1])

    return StaticFileNode(filename,
                          symlink_if_possible=_CAN_SYMLINK,
                          optimize_if_possible=optimize_if_possible,
                          context_name=context_name)