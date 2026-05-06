def DEFINE_string(flag_name, default_value, docstring, required=False):  # pylint: disable=invalid-name
    """Defines a flag of type 'string'.
    Args:
        flag_name: The name of the flag as a string.
        default_value: The default value the flag should take as a string.
        docstring: A helpful message explaining the use of the flag.
    """
    _define_helper(flag_name, default_value, docstring, str, required)