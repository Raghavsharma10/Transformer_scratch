def parse_document(text, options=0):
    """Parse a document and return the root node.

    Args:
        text (str): The text to parse.
        options (int): The cmark options.

    Returns:
        Any: Opaque reference to the root node of the parsed syntax tree.
    """
    encoded_text = text.encode('utf-8')
    return _cmark.lib.cmark_parse_document(
        encoded_text, len(encoded_text), options)