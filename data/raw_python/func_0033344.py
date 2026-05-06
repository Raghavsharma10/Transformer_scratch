def register_subparser(self, action, name, description='', arguments={}):
        '''
        Registers a new subcommand with a given function action.

        If the function action is synchronous
        '''
        action = coerce_to_synchronous(action)
        opts = []
        for flags, kwargs in arguments.items():
            if isinstance(flags, str):
                flags = tuple([flags])
            opts.append((flags, kwargs))
        self.subcommands[name] = (description, action, opts)