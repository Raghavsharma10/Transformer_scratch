def parse_config(file_path):
    """
    Convert the CISM configuration file to a python dictionary

    Args:
        file_path: absolute path to the configuration file

    Returns:
        A dictionary representation of the given file
    """
    if not os.path.isfile(file_path):
        return {}
    parser = ConfigParser()
    parser.read(file_path)
    # Strip out inline comments
    for s in parser._sections:
        for v in six.iterkeys(parser._sections[s]):
            parser._sections[s][v] = parser._sections[s][v].split("#")[0].strip()
    return parser._sections