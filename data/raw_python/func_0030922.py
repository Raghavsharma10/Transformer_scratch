def get_argument_parser():
    """Returns an argument parser object for the script."""

    desc = 'Filter FASTA file by chromosome names.'
    parser = cli.get_argument_parser(desc=desc)

    parser.add_argument(
        '-f', '--fasta-file', default='-', type=str, help=textwrap.dedent("""\
                Path of the FASTA file. The file may be gzip'ed.
                If set to ``-``, read from ``stdin``."""))

    parser.add_argument(
        '-s', '--species', type=str,
        choices=sorted(ensembl.SPECIES_CHROMPAT.keys()),
        default='human', help=textwrap.dedent("""\
            Species for which to extract genes. (This parameter is ignored
            if ``--chromosome-pattern`` is specified.)""")
    )

    parser.add_argument(
        '-c', '--chromosome-pattern', type=str, required=False,
        default=None, help=textwrap.dedent("""\
            Regular expression that chromosome names have to match.
            If not specified, determine pattern based on the setting of
            ``--species``.""")
    )

    parser.add_argument(
        '-o', '--output-file', type=str, required=True,
        help=textwrap.dedent("""\
            Path of output file. If set to ``-``, print to ``stdout``,
            and redirect logging messages to ``stderr``."""))

    parser = cli.add_reporting_args(parser)
    
    return parser