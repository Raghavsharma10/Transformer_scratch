def __parse_main(self, args):
        """Parse the main arguments only. This is a work around for python 2.7
        because argparse does not allow to parse arguments without subparsers
        """
        if six.PY2:
            self._subparsers_action.add_parser("__dummy")
            return super(FuncArgParser, self).parse_known_args(
                list(args) + ['__dummy'])
        return super(FuncArgParser, self).parse_known_args(args)