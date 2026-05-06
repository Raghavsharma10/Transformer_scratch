def escape_identifier(text, reg=KWD_RE):
    """Escape partial C identifiers so they can be used as
    attributes/arguments"""

    # see http://docs.python.org/reference/lexical_analysis.html#identifiers
    if not text:
        return "_"
    if text[0].isdigit():
        text = "_" + text
    return reg.sub(r"\1_", text)