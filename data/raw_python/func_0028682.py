def convert(input, output=None, options={}):
    """Converts a supported ``input_format`` (*oldhepdata*, *yaml*)
    to a supported ``output_format`` (*csv*, *root*, *yaml*, *yoda*).

    :param input: location of input file for *oldhepdata* format or input directory for *yaml* format
    :param output: location of output directory to which converted files will be written
    :param options: additional options such as ``input_format`` and ``output_format`` used for conversion
    :type input: str
    :type output: str
    :type options: dict
    :raise ValueError: raised if no ``input_format`` or ``output_format`` is specified
    """

    if 'input_format' not in options and 'output_format' not in options:
        raise ValueError("no input_format and output_format specified!")

    input_format = options.get('input_format', 'yaml')
    output_format = options.get('output_format', 'yaml')

    parser = Parser.get_concrete_class(input_format)(**options)
    writer = Writer.get_concrete_class(output_format)(**options)

    if not output and not writer.single_file_output:
        raise ValueError("this output_format requires specifying 'output' argument")

    # if no output was specified create proxy output to which writer can insert data
    _output = output
    if not _output:
        _output = StringIO.StringIO()

    writer.write(parser.parse(input), _output)

    # if no output was specified return output
    if not output:
        return _output.getvalue()