def generate(data, format="auto"):
    """Converts input chemical formats to json and optimizes structure.

    Args:
        data: A string or file representing a chemical
        format: The format of the `data` variable (default is 'auto')

    The `format` can be any value specified by Open Babel
    (http://openbabel.org/docs/2.3.1/FileFormats/Overview.html). The 'auto'
    option uses the extension for files (ie. my_file.mol -> mol) and defaults
    to SMILES (smi) for strings.
    """
    # Support both files and strings and attempt to infer file type
    try:
        with open(data) as in_file:
            if format == 'auto':
                format = data.split('.')[-1]
            data = in_file.read()
    except:
        if format == 'auto':
            format = 'smi'
    return format_converter.convert(data, format, 'json')