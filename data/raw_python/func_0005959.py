def parser_feed(parser, text):
    """Direct wrapper over cmark_parser_feed."""
    encoded_text = text.encode('utf-8')
    return _cmark.lib.cmark_parser_feed(
        parser, encoded_text, len(encoded_text))