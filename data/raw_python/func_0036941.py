def _get_parser(description):
    """Build an ArgumentParser with common arguments for both operations."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('key', help="Camellia key.")
    parser.add_argument('input_file', nargs='*',
                        help="File(s) to read as input data. If none are "
                        "provided, assume STDIN.")
    parser.add_argument('-o', '--output_file',
                        help="Output file. If not provided, assume STDOUT.")
    parser.add_argument('-l', '--keylen', type=int, default=128,
                        help="Length of 'key' in bits, must be in one of %s "
                        "(default 128)." % camcrypt.ACCEPTABLE_KEY_LENGTHS)
    parser.add_argument('-H', '--hexkey', action='store_true',
                        help="Treat 'key' as a hex string rather than binary.")

    return parser