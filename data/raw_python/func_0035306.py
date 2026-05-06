def add(self, *matches, **kw):  # kw=default=None, boolean=False
        '''Add an argument; this is optional, and mostly useful for setting up aliases or setting boolean=True

        Apparently `def add(self, *matches, default=None, boolean=False):` is invalid syntax in Python. Not only is this absolutely ridiculous, but the alternative `def add(self, default=None, boolean=False, *matches):` does not do what you would expect. This syntax works as intended in Python 3.

        If you provide multiple `matches` that are not dash-prefixed, only the first will be used as a positional argument.

        Specifying any positional arguments and then using `boolean=True` is just weird, and their will be no special consideration for boolean=True in that case for the position-enabled argument.
        '''
        # python syntax hack
        default = kw.get('default', None)
        boolean = kw.get('boolean', False)
        del kw
        # do not use kw after this line! It's a hack; it should never have been there in the first place.
        positional = None
        names = []
        for match in matches:
            if match.startswith('--'):
                names.append(match[2:])
            elif match.startswith('-'):
                names.append(match[1:])
            elif positional:
                # positional has already been filled
                names.append(match)
            else:
                # first positional: becomes canonical positional
                positional = match
                names.append(match)

        argument = BooleanArgument(names, default, boolean, positional)
        self.arguments.append(argument)

        # chainable
        return self