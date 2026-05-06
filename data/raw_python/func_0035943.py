def cli():
    """Parse options from the command line"""
    parser = argparse.ArgumentParser(prog="sphinx-serve",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler="resolve",
                                     description=__doc__
                                     )

    parser.add_argument("-v", "--version", action="version",
                        version="%(prog)s {0}".format(__version__)
                        )

    parser.add_argument("-h", "--host", action="store",
                        default="0.0.0.0",
                        help="Listen to the given hostname"
                        )

    parser.add_argument("-p", "--port", action="store",
                        type=int, default=8081,
                        help="Listen to given port"
                        )

    parser.add_argument("-b", "--build", action="store",
                        default="_build",
                        help="Build folder name"
                        )

    parser.add_argument("-s", "--single", action="store_true",
                        help="Serve the single-html documentation version"
                        )

    return parser.parse_args()