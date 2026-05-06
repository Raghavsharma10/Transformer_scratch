def parseArgs():
    """Read arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-names", "-n", help=".txt file of taxonomic names")
    parser.add_argument("-datasource", "-d", help="taxonomic datasource by \
which names will be resolved (default NCBI)")
    parser.add_argument("-taxonid", "-t", help="parent taxonomic ID")
    parser.add_argument("--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument('--details', help='display information about the \
program', action='store_true')
    return parser.parse_args()