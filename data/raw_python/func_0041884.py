def _parse2subparser_funcs(self, kws):
        """
        Recursive function to parse arguments to chained parsers
        """
        choices = getattr(self._subparsers_action, 'choices', {})
        replaced = {key.replace('-', '_'): key for key in choices}
        sp_commands = set(replaced).intersection(kws)
        if not sp_commands:
            if self._setup_as is not None:
                func = kws.pop(self._setup_as)
            else:
                try:
                    func = self._used_functions[-1]
                except IndexError:
                    return None
            return func(**{
                key: kws[key] for key in set(kws).difference(choices)})
        else:
            ret = {}
            for key in sp_commands:
                ret[key.replace('-', '_')] = \
                    choices[replaced[key]]._parse2subparser_funcs(
                        vars(kws[key]))
            return Namespace(**ret)