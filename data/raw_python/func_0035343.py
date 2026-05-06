def parse(self, complete=True, args=None):
        '''Parse a list of arguments, returning a dict

        See BooleanParser.parse for `args`-related documentation.

        If `complete` is True and there are values in `args` that don't have corresponding arguments,
        or there are required arguments that don't have args, then raise an error.
        '''
        opts = dict()
        positions = [argument.positional for argument in self.arguments if argument.positional]

        if args is None:
            import sys
            # skip over the program name with the [1:] slice
            args = sys.argv[1:]

        # arglist is a tuple of (is_flag, name) pairs
        arglist = peekable(parse_tokens(args))
        for is_flag, name in arglist:
            if is_flag is True:
                argument = self.find_argument(name)

                # .peek will return the default argument iff there are no more entries
                next_is_flag, next_name = arglist.peek(default=(None, None))
                # next_is_flag will be None if there are no more items, but True/False if there is a next item

                # if this argument looks for a subsequent (is set as boolean), and the subsequent is not a flag, consume it
                if argument.boolean is False and next_is_flag is False:
                    opts[name] = next_name
                    # finally, advance our iterator, but since we already have the next values, just discard it
                    arglist.next()
                else:
                    # if there is no next, or the next thing is a flag all the boolean=False's in the world can't save you then
                    opts[name] = True
            else:
                # add positional argument
                if len(positions) > 0:
                    # we pop the positions off from the left
                    position = positions.pop(0)
                    opts[position] = name
                else:
                    # the rest of the args now end up as a list in '_'
                    opts.setdefault('_', []).append(name)

        # propagate aliases and defaults:
        for argument in self.arguments:
            # merge provided value from aliases
            for name in argument.names:
                if name in opts:
                    value = opts[name]
                    # we simply break on the first match.
                    break
            else:
                # if we iterate through all names and fine none in opts, use the default
                value = argument.default

            for name in argument.names:
                opts[name] = value

        return opts