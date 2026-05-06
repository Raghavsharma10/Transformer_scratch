def get_argument_parser():
    """Creates the argument parser for the extract_entrez2gene.py script.

    Returns
    -------
    A fully configured `argparse.ArgumentParser` object.

    Notes
    -----
    This function is used by the `sphinx-argparse` extension for sphinx.

    """

    desc = 'Generate a mapping of Entrez IDs to gene symbols.'

    parser = cli.get_argument_parser(desc=desc)

    parser.add_argument(
        '-f', '--gene2acc-file', type=str, required=True,
        help=textwrap.dedent("""\
            Path of gene2accession.gz file (from
            ftp://ftp.ncbi.nlm.nih.gov/gene/DATA), or a filtered version
            thereof.""")
    )

    parser.add_argument(
        '-o', '--output-file', type=str, required=True,
        help=textwrap.dedent("""\
            Path of output file. If set to ``-``, print to ``stdout``,
            and redirect logging messages to ``stderr``.""")
    )

    parser.add_argument(
        '-l', '--log-file', type=str, default=None,
        help='Path of log file. If not specified, print to stdout.'
    )

    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress all output except warnings and errors.'
    )

    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose output. Ignored if ``--quiet`` is specified.'
    )

    return parser