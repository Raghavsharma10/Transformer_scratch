def _strip_dollars_fast(text):
    """
    Replace `$$` with `$`. raise immediately
    if `$` starting an interpolated expression is found.
    @param text: the source text
    @return: the text with dollars replaced, or raise
        HasExprException if there are interpolated expressions
    """

    def _sub(m):
        if m.group(0) == '$$':
            return '$'

        raise HasExprException()

    return _dollar_strip_re.sub(_sub, text)