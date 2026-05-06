def addFASTAEditingCommandLineOptions(parser):
    """
    Add standard FASTA editing command-line options to an argparse parser.

    These are options that can be used to alter FASTA records, NOT options
    that simply select or reject those things (for those see
    addFASTAFilteringCommandLineOptions).

    @param parser: An C{argparse.ArgumentParser} instance.
    """
    # A mutually exclusive group for --keepSites, --keepSitesFile,
    # --removeSites, and --removeSitesFile.
    group = parser.add_mutually_exclusive_group()

    # In the 4 options below, the 'indices' alternate names are kept for
    # backwards compatibility.
    group.add_argument(
        '--keepSites', '--keepIndices',
        help=('Specify 1-based sequence sites to keep. All other sites will '
              'be removed. The sites must be given in the form e.g., '
              '24,100-200,260. Note that the requested sites will be taken '
              'from the input sequences in order, not in the order given by '
              '--keepSites. I.e., --keepSites 5,8-10 will get you the same '
              'result as --keepSites 8-10,5.'))

    group.add_argument(
        '--keepSitesFile', '--keepIndicesFile',
        help=('Specify a file containing 1-based sites to keep. All other '
              'sequence sites will be removed. Lines in the file must be '
              'given in the form e.g., 24,100-200,260. See --keepSites for '
              'more detail.'))

    group.add_argument(
        '--removeSites', '--removeIndices',
        help=('Specify 1-based sites to remove. All other sequence sites will '
              'be kept. The sites must be given in the form e.g., '
              '24,100-200,260. See --keepSites for more detail.'))

    group.add_argument(
        '--removeSitesFile', '--removeIndicesFile',
        help=('Specify a file containing 1-based sites to remove. All other '
              'sequence sites will be kept. Lines in the file must be given '
              'in the form e.g., 24,100-200,260. See --keepSites for more '
              'detail.'))

    parser.add_argument(
        '--removeGaps', action='store_true', default=False,
        help="If True, gap ('-') characters in sequences will be removed.")

    parser.add_argument(
        '--truncateTitlesAfter',
        help=('A string that sequence titles (ids) will be truncated beyond. '
              'If the truncated version of a title has already been seen, '
              'that title will be skipped.'))

    parser.add_argument(
        '--removeDescriptions', action='store_true', default=False,
        help=('Read id descriptions will be removed. The '
              'description is the part of a sequence id after the '
              'first whitespace (if any).'))

    parser.add_argument(
        '--idLambda', metavar='LAMBDA-FUNCTION',
        help=('A one-argument function taking and returning a read id. '
              'E.g., --idLambda "lambda id: id.split(\'_\')[0]" or '
              '--idLambda "lambda id: id[:10]". If the function returns None, '
              'the read will be filtered out.'))

    parser.add_argument(
        '--readLambda', metavar='LAMBDA-FUNCTION',
        help=('A one-argument function taking and returning a read. '
              'E.g., --readLambda "lambda r: Read(r.id.split(\'_\')[0], '
              'r.sequence.strip(\'-\')". Make sure to also modify the quality '
              'string if you change the length of a FASTQ sequence. If the '
              'function returns None, the read will be filtered out. The '
              'function will be passed to eval with the dark.reads classes '
              'Read, DNARead, AARead, etc. all in scope.'))

    parser.add_argument(
        '--reverse', action='store_true', default=False,
        help=('Reverse the sequences. Note that this is NOT reverse '
              'complementing.'))

    parser.add_argument(
        '--reverseComplement', action='store_true', default=False,
        help='Reverse complement the sequences.')