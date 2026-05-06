def remove_unsafe_chars(text):
    """Remove unsafe unicode characters from a piece of text."""
    if isinstance(text, six.string_types):
        text = UNSAFE_RE.sub('', text)
    return text