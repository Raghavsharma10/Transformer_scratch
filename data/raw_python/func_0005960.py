def render_html(root, options=0, extensions=None):
    """Render a given syntax tree as HTML.

    Args:
        root (Any): The reference to the root node of the syntax tree.
        options (int): The cmark options.
        extensions (Any): The reference to the syntax extensions, generally
            from :func:`parser_get_syntax_extensions`

    Returns:
        str: The rendered HTML.
    """
    if extensions is None:
        extensions = _cmark.ffi.NULL

    raw_result = _cmark.lib.cmark_render_html(
        root, options, extensions)

    return _cmark.ffi.string(raw_result).decode('utf-8')