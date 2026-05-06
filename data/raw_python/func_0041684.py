def add_argparser(self, root, parents):
        """
        Add arguments for this command.
        """
        parser = root.add_parser('report', parents=parents)
        parser.set_defaults(func=self)

        parser.add_argument(
            '--auth',
            dest='auth', action='store',
            help='Path to the authorized credentials file (analytics.dat).'
        )

        parser.add_argument(
            '--title',
            dest='title', action='store',
            help='User-friendly title for your report.'
        )

        parser.add_argument(
            '--property-id',
            dest='property-id', action='store',
            help='Google Analytics ID of the property to query.'
        )

        parser.add_argument(
            '--start-date',
            dest='start-date', action='store',
            help='Start date for the query in YYYY-MM-DD format.'
        )

        parser.add_argument(
            '--end-date',
            dest='end-date', action='store',
            help='End date for the query in YYYY-MM-DD format. Supersedes --ndays.'
        )

        parser.add_argument(
            '--ndays',
            dest='ndays', action='store', type=int,
            help='The number of days from the start-date to query. Requires start-date. Superseded by end-date.'
        )

        parser.add_argument(
            '--domain',
            dest='domain', action='store',
            help='Restrict results to only urls with this domain.'
        )

        parser.add_argument(
            '--prefix',
            dest='prefix', action='store',
            help='Restrict results to only urls with this prefix.'
        )

        parser.add_argument(
            'input_path',
            action='store',
            help='Path to either a YAML configuration file or pre-reported JSON data.'
        )

        parser.add_argument(
            'output_path',
            action='store',
            help='Path to output either an HTML report or a JSON data file.'
        )

        return parser