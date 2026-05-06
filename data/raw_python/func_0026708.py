def _add_parser_arguments_analyze(self, subparsers):
        """Create a parser for the 'analyze' subcommand.
        """
        lyze_pars = subparsers.add_parser(
            "analyze",
            help="Perform basic analysis on this catalog.")

        lyze_pars.add_argument(
            '--count', '-c', dest='count',
            default=False, action='store_true',
            help='Determine counts of entries, files, etc.')

        return lyze_pars