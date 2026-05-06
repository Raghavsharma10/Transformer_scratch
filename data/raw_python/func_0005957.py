def markdown_to_html_with_extensions(text, options=0, extensions=None):
    """Render the given text to Markdown, using extensions.

    This is a high-level wrapper over the various functions needed to enable
    extensions, attach them to a parser, and render html.

    Args:
        text (str): The text to render to Markdown.
        options (int): The cmark options.
        extensions (Sequence[str]): The list of extension names to use.

    Returns:
        str: The rendered markdown.
    """
    if extensions is None:
        extensions = []

    core_extensions_ensure_registered()

    cmark_extensions = []
    for extension_name in extensions:
        extension = find_syntax_extension(extension_name)
        if extension is None:
            raise ValueError('Unknown extension {}'.format(extension_name))
        cmark_extensions.append(extension)

    parser = parser_new(options=options)

    try:
        for extension in cmark_extensions:
            parser_attach_syntax_extension(parser, extension)

        parser_feed(parser, text)

        root = parser_finish(parser)

        if _cmark.lib.cmark_node_get_type(root) == _cmark.lib.CMARK_NODE_NONE:
            raise ValueError('Error parsing markdown!')

        extensions_ll = parser_get_syntax_extensions(parser)

        output = render_html(root, options=options, extensions=extensions_ll)

    finally:
        parser_free(parser)

    return output