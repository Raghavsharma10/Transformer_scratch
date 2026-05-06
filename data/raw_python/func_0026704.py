def load_args(self, args, clargs):
        """Parse arguments and return configuration settings.
        """
        # Parse All Arguments
        args = self.parser.parse_args(args=clargs, namespace=args)

        # Print the help information if no subcommand is given
        # subcommand is required for operation
        if args.subcommand is None:
            self.parser.print_help()
            args = None

        return args