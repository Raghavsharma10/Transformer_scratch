def generate_string(converter, input, format='xml'):
    """
    Like generate(), but reads the input from a string instead of
    from a file.

    :type  converter: compiler.Context
    :param converter: The compiled converter.
    :type  input: str
    :param input: The string to convert.
    :type  format: str
    :param format: The output format.
    :rtype:  str
    :return: The resulting output.
    """
    serializer = generator.new(format)
    if serializer is None:
        raise TypeError('invalid output format ' + repr(format))
    builder = Builder()
    converter.parse_string(input, builder)
    return builder.serialize(serializer)