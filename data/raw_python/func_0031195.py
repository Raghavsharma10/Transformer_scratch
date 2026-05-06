def addFilteringOptions(parser, samfileIsPositionalArg=False):
        """
        Add options to an argument parser for filtering SAM/BAM.

        @param samfileIsPositionalArg: If C{True} the SAM/BAM file must
            be given as the final argument on the command line (without
            being preceded by --samfile).
        @param parser: An C{argparse.ArgumentParser} instance.
        """
        parser.add_argument(
            '%ssamfile' % ('' if samfileIsPositionalArg else '--'),
            required=True,
            help='The SAM/BAM file to filter.')

        parser.add_argument(
            '--referenceId', metavar='ID', nargs='+', action='append',
            help=('A reference sequence id whose alignments should be kept '
                  '(alignments against other references will be dropped). '
                  'If omitted, alignments against all references will be '
                  'kept. May be repeated.'))

        parser.add_argument(
            '--dropUnmapped', default=False, action='store_true',
            help='If given, unmapped matches will not be output.')

        parser.add_argument(
            '--dropSecondary', default=False, action='store_true',
            help='If given, secondary matches will not be output.')

        parser.add_argument(
            '--dropSupplementary', default=False, action='store_true',
            help='If given, supplementary matches will not be output.')

        parser.add_argument(
            '--dropDuplicates', default=False, action='store_true',
            help=('If given, matches flagged as optical or PCR duplicates '
                  'will not be output.'))

        parser.add_argument(
            '--keepQCFailures', default=False, action='store_true',
            help=('If given, reads that are considered quality control '
                  'failures will be included in the output.'))

        parser.add_argument(
            '--minScore', type=float,
            help=('If given, alignments with --scoreTag (default AS) values '
                  'less than this value will not be output. If given, '
                  'alignments that do not have a score will not be output.'))

        parser.add_argument(
            '--maxScore', type=float,
            help=('If given, alignments with --scoreTag (default AS) values '
                  'greater than this value will not be output. If given, '
                  'alignments that do not have a score will not be output.'))

        parser.add_argument(
            '--scoreTag', default='AS',
            help=('The alignment tag to extract for --minScore and --maxScore '
                  'comparisons.'))