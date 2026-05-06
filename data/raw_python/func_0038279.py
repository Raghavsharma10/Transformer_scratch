def add_sam2rnf_parser(subparsers, subcommand, help, description, simulator_name=None):
    """Add another parser for a SAM2RNF-like command.

	Args:
		subparsers (subparsers): File name of the genome from which read tuples are created (FASTA file).
		simulator_name (str): Name of the simulator used in comments.
	"""

    parser_sam2rnf = subparsers.add_parser(subcommand, help=help, description=description)

    parser_sam2rnf.set_defaults(func=sam2rnf)

    parser_sam2rnf.add_argument(
        '-s', '--sam', type=str, metavar='file', dest='sam_fn', required=True,
        help='Input SAM/BAM with true (expected) alignments of the reads  (- for standard input).'
    )

    _add_shared_params(parser_sam2rnf, unmapped_switcher=True)

    parser_sam2rnf.add_argument(
        '-n',
        '--simulator-name',
        type=str,
        metavar='str',
        dest='simulator_name',
        default=simulator_name,
        help='Name of the simulator (for RNF).' if simulator_name is not None else argparse.SUPPRESS,
    )