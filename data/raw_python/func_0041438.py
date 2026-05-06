def add_argparser(self, root, parents):
        """
        Add arguments for this command.
        """
        parents.append(tools.argparser)

        parser = root.add_parser('auth', parents=parents)
        parser.set_defaults(func=self)

        parser.add_argument(
            '--secrets',
            dest='secrets', action='store',
            help='Path to the authorization secrets file (client_secrets.json).'
        )

        return parser