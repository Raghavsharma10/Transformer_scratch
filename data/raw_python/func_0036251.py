def DEFINE_boolean(flag_name, default_value, docstring):  # pylint: disable=invalid-name
    """Defines a flag of type 'boolean'.
    Args:
        flag_name: The name of the flag as a string.
        default_value: The default value the flag should take as a boolean.
        docstring: A helpful message explaining the use of the flag.
    """

    # Register a custom function for 'bool' so --flag=True works.
    def str2bool(bool_str):
        """Return a boolean value from a give string."""
        return bool_str.lower() in ('true', 't', '1')

    get_context_parser().add_argument(
        '--' + flag_name,
        nargs='?',
        const=True,
        help=docstring,
        default=default_value,
        type=str2bool)

    # Add negated version, stay consistent with argparse with regard to
    # dashes in flag names.
    get_context_parser().add_argument(
        '--no' + flag_name,
        action='store_false',
        dest=flag_name.replace('-', '_'))