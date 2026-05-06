def peep_hash(argv):
    """Return the peep hash of one or more files, returning a shell status code
    or raising a PipException.

    :arg argv: The commandline args, starting after the subcommand

    """
    parser = OptionParser(
        usage='usage: %prog hash file [file ...]',
        description='Print a peep hash line for one or more files: for '
                    'example, "# sha256: '
                    'oz42dZy6Gowxw8AelDtO4gRgTW_xPdooH484k7I5EOY".')
    _, paths = parser.parse_args(args=argv)
    if paths:
        for path in paths:
            print('# sha256:', hash_of_file(path))
        return ITS_FINE_ITS_FINE
    else:
        parser.print_usage()
        return COMMAND_LINE_ERROR