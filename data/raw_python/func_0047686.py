def _add_opt_argument(self, opt_args, arg_parser):
        """Add an argument to an instantiated parser.

        :param opt_args: ``dict``
        :param arg_parser: ``object``
        """
        option_args = opt_args.copy()

        groups = option_args.pop('groups', None)
        if groups:
            self._add_group(
                parser=arg_parser,
                groups=groups,
                option_args=option_args
            )

        exclusive_args = option_args.pop('mutually_exclusive', None)
        if exclusive_args:
            self._add_mutually_exclusive_group(
                parser=arg_parser,
                groups=exclusive_args,
                option_args=option_args
            )

        for k, v in option_args.items():
            self._add_arg(parser=arg_parser, value_dict=v)