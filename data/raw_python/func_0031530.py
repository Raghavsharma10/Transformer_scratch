def get_argument_parser():
    """Function to obtain the argument parser.

    Returns
    -------
    A fully configured `argparse.ArgumentParser` object.

    Notes
    -----
    This function is used by the `sphinx-argparse` extension for sphinx.

    """
    file_mv = cli.file_mv

    desc = 'Extracts gene-level expression data from StringTie output.'
    parser = cli.get_argument_parser(desc)

    parser.add_argument(
        '-s', '--stringtie-file', type=str, required=True, metavar=file_mv,
        help="""Path of the StringTie output file ."""
    )

    parser.add_argument(
        '-g', '--gene-file', type=str, required=True, metavar=file_mv,
        help="""File containing a list of protein-coding genes."""
    )

    parser.add_argument(
        '--no-novel-transcripts', action='store_true',
        help="""Ignore novel transcripts."""
    )

    # parser.add_argument(
    #     '--ambiguous-transcripts', default = 'ignore',
    #      help='Strategy for counting expression of ambiguous novel '
    #            'transcripts.'
    # )
    # possible strategies for ambiguous transcripts: 'ignore','highest','all'

    parser.add_argument(
        '-o', '--output-file', type=str, required=True, metavar=file_mv,
        help="""Path of output file."""
    )

    cli.add_reporting_args(parser)

    return parser