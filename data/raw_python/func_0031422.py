def addFASTAFilteringCommandLineOptions(parser):
    """
    Add standard FASTA filtering command-line options to an argparse parser.

    These are options that can be used to select or omit entire FASTA records,
    NOT options that change them (for that see
    addFASTAEditingCommandLineOptions).

    @param parser: An C{argparse.ArgumentParser} instance.
    """
    parser.add_argument(
        '--minLength', type=int,
        help='The minimum sequence length')

    parser.add_argument(
        '--maxLength', type=int,
        help='The maximum sequence length')

    parser.add_argument(
        '--whitelist', action='append',
        help='Sequence titles (ids) that should be whitelisted')

    parser.add_argument(
        '--blacklist', action='append',
        help='Sequence titles (ids) that should be blacklisted')

    parser.add_argument(
        '--whitelistFile',
        help=('The name of a file that contains sequence titles (ids) that '
              'should be whitelisted, one per line'))

    parser.add_argument(
        '--blacklistFile',
        help=('The name of a file that contains sequence titles (ids) that '
              'should be blacklisted, one per line'))

    parser.add_argument(
        '--titleRegex', help='A regex that sequence titles (ids) must match.')

    parser.add_argument(
        '--negativeTitleRegex',
        help='A regex that sequence titles (ids) must not match.')

    # A mutually exclusive group for --keepSequences and --removeSequences.
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        '--keepSequences',
        help=('Specify (1-based) ranges of sequence numbers that should be '
              'kept. E.g., --keepSequences 1-3,5 will output just the 1st, '
              '2nd, 3rd, and 5th sequences. All others will be omitted.'))

    group.add_argument(
        '--removeSequences',
        help=('Specify (1-based) ranges of sequence numbers that should be '
              'removed. E.g., --removeSequences 1-3,5 will output all but the '
              '1st, 2nd, 3rd, and 5th sequences. All others will be ouput.'))

    parser.add_argument(
        '--head', type=int, metavar='N',
        help='Only the first N sequences will be printed.')

    parser.add_argument(
        '--removeDuplicates', action='store_true', default=False,
        help=('Duplicate reads will be removed, based only on '
              'sequence identity. The first occurrence is kept.'))

    parser.add_argument(
        '--removeDuplicatesById', action='store_true', default=False,
        help=('Duplicate reads will be removed, based only on '
              'read id. The first occurrence is kept.'))

    # See the docstring for dark.reads.Reads.filter for more detail on
    # randomSubset.
    parser.add_argument(
        '--randomSubset', type=int,
        help=('An integer giving the number of sequences that should be kept. '
              'These will be selected at random.'))

    # See the docstring for dark.reads.Reads.filter for more detail on
    # trueLength.
    parser.add_argument(
        '--trueLength', type=int,
        help=('The number of reads in the FASTA input. Only to be used with '
              'randomSubset'))

    parser.add_argument(
        '--sampleFraction', type=float,
        help=('A [0.0, 1.0] C{float} indicating a fraction of the reads that '
              'should be allowed to pass through the filter. The sample size '
              'will only be approximately the product of the sample fraction '
              'and the number of reads. The sample is taken at random.'))

    parser.add_argument(
        '--sequenceNumbersFile',
        help=('A file of (1-based) sequence numbers to retain. Numbers must '
              'be one per line.'))