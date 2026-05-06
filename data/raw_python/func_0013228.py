def generate_to_file(converter,
                     input_file,
                     output_file,
                     format='xml',
                     in_encoding='utf8',
                     out_encoding='utf8'):
    """
    Like generate(), but writes the output to the given output file
    instead.

    :type  converter: compiler.Context
    :param converter: The compiled converter.
    :type  input_file: str
    :param input_file: Name of a file to convert.
    :type  output_file: str
    :param output_file: The output filename.
    :type  format: str
    :param format: The output format.
    :type  in_encoding: str
    :param in_encoding: Character encoding of the input file.
    :type  out_encoding: str
    :param out_encoding: Character encoding of the output file.
    :rtype:  str
    :return: The resulting output.
    """
    with codecs.open(output_file, 'w', encoding=out_encoding) as thefile:
        result = generate(converter, input_file, format=format, encoding=in_encoding)
        thefile.write(result)