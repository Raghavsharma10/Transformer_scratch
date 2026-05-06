def arg_parser(self, passed_args=None):
        """Setup argument Parsing.

        If preset args are to be specified use the ``passed_args`` tuple.

        :param passed_args: ``list``
        :return: ``dict``
        """
        parser, subpar, remaining_argv = self._setup_parser()
        if not isinstance(passed_args, list):
            passed_args = list()

        # Extend the passed args with the remaining parsed args
        if remaining_argv:
            passed_args.extend(remaining_argv)

        optional_args = self.arguments.get('optional_args')
        if optional_args:
            self._add_opt_argument(opt_args=optional_args, arg_parser=parser)

        subparsed_args = self.arguments.get('subparsed_args')
        if subparsed_args:
            for argument, value in subparsed_args.items():
                if 'optional_args' in value:
                    optional_args = value.pop('optional_args')
                else:
                    optional_args = dict()

                if 'shared_args' in value:
                    set_shared_args = self.arguments.get('shared_args')
                    _shared_args = value.pop('shared_args', list())
                    for shared_arg in _shared_args:
                        optional_args[shared_arg] = set_shared_args[shared_arg]

                action = subpar.add_parser(
                    argument,
                    **value
                )
                action.set_defaults(parsed_command=argument)

                if optional_args:
                    self._add_opt_argument(
                        opt_args=optional_args, arg_parser=action
                    )

        positional_args = self.arguments.get('positional_args')
        if positional_args:
            for argument in positional_args.keys():
                _arg = positional_args[argument]

                parser.add_argument(
                    argument,
                    **_arg
                )

        # Return the parsed arguments as a dict
        return vars(parser.parse_args(args=passed_args))