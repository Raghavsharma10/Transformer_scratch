def get_parser_class():
    """
    Returns the parser according to the system platform
    """
    global distro
    if distro == 'Linux':
        Parser = parser.LinuxParser
        if not os.path.exists(Parser.get_command()[0]):
            Parser = parser.UnixIPParser
    elif distro in ['Darwin', 'MacOSX']:
        Parser = parser.MacOSXParser
    elif distro == 'Windows':
        # For some strange reason, Windows will always be win32, see:
        # https://stackoverflow.com/a/2145582/405682
        Parser = parser.WindowsParser
    else:
        Parser = parser.NullParser
        Log.error("Unknown distro type '%s'." % distro)
    Log.debug("Distro detected as '%s'" % distro)
    Log.debug("Using '%s'" % Parser)

    return Parser