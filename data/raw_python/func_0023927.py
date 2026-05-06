def flatatt(attrs):
    """
    Pilfered from `django.forms.utils`:
    Convert a dictionary of attributes to a single string.
    The returned string will contain a leading space followed by key="value",
    XML-style pairs. In the case of a boolean value, the key will appear
    without a value. Otherwise, the value is formatted through its own dict of `attrs`,
    which can be useful to parametrize Angular directives.
    It is assumed that the keys do not need to be
    XML-escaped. If the passed dictionary is empty, then return an empty
    string.

    The result is passed through 'mark_safe' (by way of 'format_html_join').
    """
    key_value_attrs = []
    boolean_attrs = []
    for attr, value in attrs.items():
        if isinstance(value, bool):
            if value:
                boolean_attrs.append((attr,))
        else:
            try:
                value = value.format(**attrs)
            except KeyError:
                pass
            key_value_attrs.append((attr, value))

    return (
        format_html_join('', ' {}="{}"', sorted(key_value_attrs)) +
        format_html_join('', ' {}', sorted(boolean_attrs))
    )