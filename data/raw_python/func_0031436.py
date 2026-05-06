def addFASTACommandLineOptions(parser):
    """
    Add standard command-line options to an argparse parser.

    @param parser: An C{argparse.ArgumentParser} instance.
    """

    parser.add_argument(
        '--fastaFile', type=open, default=sys.stdin, metavar='FILENAME',
        help=('The name of the FASTA input file. Standard input will be read '
              'if no file name is given.'))

    parser.add_argument(
        '--readClass', default='DNARead', choices=readClassNameToClass,
        metavar='CLASSNAME',
        help=('If specified, give the type of the reads in the input. '
              'Possible choices: %s.' % ', '.join(readClassNameToClass)))

    # A mutually exclusive group for either --fasta, --fastq, or --fasta-ss
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        '--fasta', default=False, action='store_true',
        help=('If specified, input will be treated as FASTA. This is the '
              'default.'))

    group.add_argument(
        '--fastq', default=False, action='store_true',
        help='If specified, input will be treated as FASTQ.')

    group.add_argument(
        '--fasta-ss', dest='fasta_ss', default=False, action='store_true',
        help=('If specified, input will be treated as PDB FASTA '
              '(i.e., regular FASTA with each sequence followed by its '
              'structure).'))