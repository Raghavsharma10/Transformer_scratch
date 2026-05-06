def _check_args(self, source):
        '''Validate the argument section.

        Args may be either a dict or a list (to allow multiple positional args).
        '''
        path = [source]
        args = self.parsed_yaml.get('args', {})
        self._assert_struct_type(args, 'args', (dict, list), path)
        path.append('args')
        if isinstance(args, dict):
            for argn, argattrs in args.items():
                self._check_one_arg(path, argn, argattrs)
        else: # must be list - already asserted struct type
            for argdict in args:
                self._assert_command_dict(argdict, '[list-item]', path)
                argn, argattrs = list(argdict.items())[0] # safe - length asserted on previous line
                self._check_one_arg(path, argn, argattrs)