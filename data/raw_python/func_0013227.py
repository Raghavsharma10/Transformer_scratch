def generate(converter, input_file, format='xml', encoding='utf8'):
    """
    Given a converter (as returned by compile()), this function reads
    the given input file and converts it to the requested output format.

    Supported output formats are 'xml', 'yaml', 'json', or 'none'.

    :type  converter: compiler.Context
    :param converter: The compiled converter.
    :type  input_file: str
    :param input_file: Name of a file to convert.
    :type  format: str
    :param format: The output format.
    :type  encoding: str
    :param encoding: Character encoding of the input file.
    :rtype:  str
    :return: The resulting output.
    """
    with codecs.open(input_file, encoding=encoding) as thefile:
        return generate_string(converter, thefile.read(), format=format)