def _str_to_list(value, separator):
    """Convert a string to a list with sanitization."""
    value_list = [item.strip() for item in value.split(separator)]
    value_list_sanitized = builtins.list(filter(None, value_list))
    if len(value_list_sanitized) > 0:
        return value_list_sanitized
    else:
        raise ValueError('Invalid list variable.')