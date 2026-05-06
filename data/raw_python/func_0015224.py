def get_gui_hint(self, hint):
        """Returns the value for specified gui hint (or a sensible default value,
        if this argument doesn't specify the hint).

        Args:
            hint: name of the hint to get value for
        Returns:
            value of the hint specified in yaml or a sensible default
        """
        if hint == 'type':
            # 'self.kwargs.get('nargs') == 0' is there for default_iff_used, which may
            # have nargs: 0, so that it works similarly to 'store_const'
            if self.kwargs.get('action') == 'store_true' or self.kwargs.get('nargs') == 0:
                return 'bool'
            # store_const is represented by checkbox, but computes default differently
            elif self.kwargs.get('action') == 'store_const':
                return 'const'
            return self.gui_hints.get('type', 'str')
        elif hint == 'default':
            hint_type = self.get_gui_hint('type')
            hint_default = self.gui_hints.get('default', None)
            arg_default = self.kwargs.get('default', None)
            preserved_value = None
            if 'preserved' in self.kwargs:
                preserved_value = config_manager.get_config_value(self.kwargs['preserved'])

            if hint_type == 'path':
                if preserved_value is not None:
                    default = preserved_value
                elif hint_default is not None:
                    default = hint_default.replace('$(pwd)', utils.get_cwd_or_homedir())
                else:
                    default = arg_default or '~'
                return os.path.abspath(os.path.expanduser(default))
            elif hint_type == 'bool':
                return hint_default or arg_default or False
            elif hint_type == 'const':
                return hint_default or arg_default
            else:
                if hint_default == '$(whoami)':
                    hint_default = getpass.getuser()
                return preserved_value or hint_default or arg_default or ''