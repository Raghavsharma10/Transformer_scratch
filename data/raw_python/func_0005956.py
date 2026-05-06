def markdown_to_html(text, options=0):
    """Render the given text to Markdown.

    This is a direct interface to ``cmark_markdown_to_html``.

    Args:
        text (str): The text to render to Markdown.
        options (int): The cmark options.

    Returns:
        str: The rendered markdown.
    """
    encoded_text = text.encode('utf-8')
    raw_result = _cmark.lib.cmark_markdown_to_html(
        encoded_text, len(encoded_text), options)
    return _cmark.ffi.string(raw_result).decode('utf-8')