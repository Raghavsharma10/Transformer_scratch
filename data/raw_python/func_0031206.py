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

    desc = 'Find all runs (SRR..) associated with an SRA experiment (SRX...).'

    parser = cli.get_argument_parser(desc=desc)

    parser.add_argument(
        '-e', '--experiment-file', type=str, required=True, metavar=file_mv,
        help='File with SRA experiment IDs (starting with "SRX").'
    )

    parser.add_argument(
        '-o', '--output-file', type=str, required=True, metavar=file_mv,
        help='The output file.'
    )

    cli.add_reporting_args(parser)

    return parser