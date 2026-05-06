def t_QUOTED_STRING(t):
    r'"([^"\\]|\\["\\])*"'
    # TODO: Add support for:
    # - An undefined escape sequence (such as "\a" in a context where "a"
    # has no special meaning) is interpreted as if there were no backslash
    # (in this case, "\a" is just "a"), though that may be changed by
    # extensions.
    # - Non-printing characters such as tabs, CRLF, and control characters
    # are permitted in quoted strings.  Quoted strings MAY span multiple
    # lines.  An unencoded NUL (US-ASCII 0) is not allowed in strings.
    t.value = t.value.strip('"').replace(r'\"', '"').replace(r'\\', '\\')
    return t