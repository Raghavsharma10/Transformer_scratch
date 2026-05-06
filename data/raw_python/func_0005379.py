def convert_boolean(string_value):
    """Converts a string to a boolean (see CONVERTERS).
    There is a converter function for each column type.

    Boolean strings are independent of case. Values interpreted as True
    are: "yes", "true", "on", "1". values interpreted as False are
    "no", "false", "off", "0". Any other value will result in a ValueError.

    :param string_value: The string to convert

    :raises: ValueError if the string cannot be represented by a boolean
    """

    lean_string_value = string_value.strip().lower()
    if lean_string_value in ['yes', 'true', 'on', '1']:
        return True
    elif lean_string_value in ['no', 'false', 'off', '0']:
        return False

    # Not recognised boolean if we get here
    raise ValueError('Unrecognised boolean ({})'.format(lean_string_value))