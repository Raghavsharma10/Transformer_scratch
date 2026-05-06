def create_arguments(self, subparsers=False):
        """Create and add the arguments

        Parameters
        ----------
        subparsers: bool
            If True, the arguments of the subparsers are also created"""
        ret = []
        if not self._finalized:
            for arg, d in self.unfinished_arguments.items():
                try:
                    not_positional = int(not d.pop('positional', False))
                    short = d.pop('short', None)
                    long_name = d.pop('long', None)
                    if short is None and long_name is None:
                        raise ValueError(
                            "Either a short (-) or a long (--) argument must "
                            "be provided!")
                    if not not_positional:
                        short = arg
                        long_name = None
                        d.pop('dest', None)
                    if short == long_name:
                        long_name = None
                    args = []
                    if short:
                        args.append('-' * not_positional + short)
                    if long_name:
                        args.append('--' * not_positional + long_name)
                    group = d.pop('group', self)
                    if d.get('action') in ['store_true', 'store_false']:
                        d.pop('metavar', None)
                    ret.append(group.add_argument(*args, **d))
                except Exception:
                    print('Error while creating argument %s' % arg)
                    raise
        else:
            raise ValueError('Parser has already been finalized!')
        self._finalized = True
        if subparsers and self._subparsers_action is not None:
            for parser in self._subparsers_action.choices.values():
                parser.create_arguments(True)
        return ret