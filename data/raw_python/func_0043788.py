def parse_args(cls, logging=False):
        """
        Parses command line arguments.

        Looks for --liveandletdie [host]

        :returns:
            A ``(str(host), int(port))`` or ``(None, None)`` tuple.
        """

        cls._add_args()
        args = cls._argument_parser.parse_args()

        if args.liveandletdie:
            _log(logging, 'Running as test live server at {0}'
                 .format(args.liveandletdie))
            return split_host(args.liveandletdie)
        else:
            return None, None