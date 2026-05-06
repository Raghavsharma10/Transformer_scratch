def parse_args(args=None):
    """
    Handles the parsing of options for LIVVkit's command line interface

    Args:
        args: The list of arguments, typically sys.argv[1:]
    """
    parser = argparse.ArgumentParser(description="Main script to run LIVVkit.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')

    parser.add_argument('-o', '--out-dir',
                        default=os.path.join(os.getcwd(), "vv_" + time.strftime("%Y-%m-%d")),
                        help='Location to output the LIVVkit webpages.'
                        )

    parser.add_argument('-v', '--verify',
                        nargs=2,
                        default=None,
                        help=' '.join(['Specify the locations of the test and bench bundle to',
                                       'compare (respectively).'
                                       ])
                        )

    parser.add_argument('-V', '--validate',
                        action='store',
                        nargs='+',
                        default=None,
                        help=' '.join(['Specify the location of the configuration files for',
                                       'validation tests.'
                                       ])
                        )

    # FIXME: this just short-circuits to the validation option, and should become its own module
    parser.add_argument('-e', '--extension',
                        action='store',
                        nargs='+',
                        default=None,
                        dest='validate',
                        metavar='EXTENSION',
                        help=' '.join(['Specify the location of the configuration files for',
                                       'LIVVkit extensions.'
                                       ])
                        )

    parser.add_argument('-p', '--publish',
                        action='store_true',
                        help=' '.join(['Also produce a publication quality copy of the figure in',
                                       'the output directory (eps, 600d pi).'
                                       ])
                        )

    parser.add_argument('-s', '--serve',
                        nargs='?', type=int, const=8000,
                        help=' '.join(['Start a simple HTTP server for the output website specified',
                                       'by OUT_DIR on port SERVE.'
                                       ])
                        )

    parser.add_argument('--version',
                        action='version',
                        version='LIVVkit {}'.format(livvkit.__version__),
                        help="Show LIVVkit's version number and exit"
                        )

    return init(parser.parse_args(args))