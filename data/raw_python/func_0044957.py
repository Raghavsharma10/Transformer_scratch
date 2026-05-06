def main(args=sys.argv[1:]):
    """
    Command-line interface for nestagg
    """
    parser = argparse.ArgumentParser(description="""Aggregate results of
            nestly runs""")
    subparsers = parser.add_subparsers()
    delim_parser = subparsers.add_parser('delim', help="""Combine control files
            with delimited files.""")
    delim_parser.set_defaults(func=delim)
    key_group = delim_parser.add_mutually_exclusive_group()
    key_group.add_argument('-k', '--keys', help="""Comma separated list of
            keys from the JSON file to include [default: all keys]""",
            type=comma_separated_values)
    key_group.add_argument('-x', '--exclude-keys', help="""Comma separated
            list of keys from the JSON file not to include [default:
            %(default)s]""", type=comma_separated_values)
    delim_parser.add_argument('-m', '--missing-action', choices=('fail',
        'warn'), help="""Action to take when a file is missing [default:
        %(default)s]""", default='fail')
    delim_parser.add_argument('file_template', help="""Template for the
            delimited file to read in each directory [e.g. '{run_id}.csv']""")
    delim_parser.add_argument('control_files', metavar="control.json",
            help="""Control files""", nargs="*")
    delim_parser.add_argument('-d', '--directory', help="""Run on all control
            files under %(metavar)s. May be used in place of specifying control
            files.""", metavar='DIR')
    delim_parser.add_argument('-s', '--separator', default=DEFAULT_SEP,
            help="""Separator [default: %(default)s]""")
    delim_parser.add_argument('-t', '--tab', action='store_const',
            dest='separator', const='\t', help="""Files are tab-separated""")
    delim_parser.add_argument('-o', '--output', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file [default: stdout]""")

    arguments = parser.parse_args()

    arguments.func(arguments)