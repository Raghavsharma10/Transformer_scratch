def collapse_spaces(text):
    """Remove newlines, tabs and multiple spaces with single spaces."""
    if not isinstance(text, six.string_types):
        return text
    return COLLAPSE_RE.sub(WS, text).strip(WS)