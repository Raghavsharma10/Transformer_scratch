def add_command(self, factory, main, name=None, context_kwargs=None):
        """
        Attach a command directly to the :class:`CLI` object.

        """
        if name is None:
            name = factory.__name__.replace('_', '-')

        if context_kwargs is None:
            context_kwargs = {}

        short_desc, long_desc = parse_docstring(factory.__doc__)
        if long_desc:
            long_desc = short_desc + '\n\n' + long_desc

        # determine the absolute import string if relative
        if isinstance(main, str) and (
            main.startswith('.') or main.startswith(':')
        ):
            module = __import__(factory.__module__, None, None, ['__doc__'])
            package = package_for_module(module)
            if main in ['.', ':']:
                main = package.__name__
            else:
                main = package.__name__ + main

        self.commands[name] = CommandMeta(
            factory=factory,
            main=main,
            name=name,
            help=short_desc,
            description=long_desc,
            context_kwargs=context_kwargs,
        )