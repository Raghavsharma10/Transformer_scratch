def main(argv=None):
    '''
    Handles command line arguments and gets things started.

    :param argv: List of arguments, as if specified on the command-line.
                 If None, ``sys.argv[1:]`` is used instead.
    :type argv: list of str
    '''
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Takes one or more file paths and reports their detected \
                     encodings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('input',
                        help='File whose encoding we would like to determine.',
                        type=argparse.FileType('rb'), nargs='*',
                        default=[sys.stdin])
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    for f in args.input:
        if f.isatty():
            print("You are running chardetect interactively. Press " +
                  "CTRL-D twice at the start of a blank line to signal the " +
                  "end of your input. If you want help, run chardetect " +
                  "--help\n", file=sys.stderr)
        print(description_of(f, f.name))