def build_parser(parser: argparse.ArgumentParser) -> None:
    """Build a parser for CLI arguments and options."""
    parser.add_argument(
        '--delimiter',
        help='a delimiter for the samples (teeth) in the key',
        default=' ',
    )
    parser.add_argument(
        '--encoding',
        help='the encoding of the population file',
        default='utf-8',
    )
    parser.add_argument(
        '--nsamples', '-n',
        help='the number of random samples to take',
        type=int,
        default=6,
        dest='nteeth',
    )
    parser.add_argument(
        '--population', '-p',
        help='{0}, or a path to a file of line-delimited items'.format(
            ', '.join(POPULATIONS.keys()),
        ),
        default='/usr/share/dict/words',
    )
    parser.add_argument(
        '--stats',
        help='show statistics for the key',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {0}'.format(__version__),
    )