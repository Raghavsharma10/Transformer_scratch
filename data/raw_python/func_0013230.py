def generate_string_to_file(converter,
                            input,
                            output_file,
                            format='xml',
                            out_encoding='utf8'):
    """
    Like generate(), but reads the input from a string instead of
    from a file, and writes the output to the given output file.

    :type  converter: compiler.Context
    :param converter: The compiled converter.
    :type  input: str
    :param input: The string to convert.
    :type  output_file: str
    :param output_file: The output filename.
    :type  format: str
    :param format: The output format.
    :type  out_encoding: str
    :param out_encoding: Character encoding of the output file.
    :rtype:  str
    :return: The resulting output.
    """
    with codecs.open(output_file, 'w', encoding=out_encoding) as thefile:
        result = generate_string(converter, input, format=format)
        thefile.write(result)