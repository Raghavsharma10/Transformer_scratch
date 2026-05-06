def dispatch(cls, arguments, **kwargs):
        """Dispatch arguments parsed by docopt to the cmd with matching spec.

        :param arguments:
        :param kwargs:
        :return: exit_code
        """
        # first match wins
        # spec: all '-' elements must match, all others are False;
        #       '<sth>' elements are converted to call args on order of
        #       appearance
        #
        # kwargs are provided to dispatch call and used in func call
        for spec, func in cls._specs:
            # if command and arguments.get(command) and match(args):
            args = []  # specified args in order of appearance
            options = list(filter(lambda k: k.startswith('-') and
                                       (arguments[k] or k in spec),
                             arguments.keys()))
            cmds = list(filter(lambda k: not (k.startswith('-') or
                                         k.startswith('<')) and arguments[k],
                          arguments.keys()))
            args_spec = list(filter(lambda k: k.startswith('<'), spec))
            cmd_spec = list(filter(lambda k: not (k.startswith('-') or
                                             k.startswith('<')), spec))
            for element in spec:
                if element.startswith('-'):
                    # element is an option
                    if element in options:
                        args.append(arguments.get(element, False))
                        options.remove(element)
                elif element.startswith('<') and \
                        not arguments.get(element) is False:
                    # element is an argument
                    args.append(arguments.get(element))
                    if element in args_spec:
                        args_spec.remove(element)
                else:
                    # element is a command
                    if element in cmds and element in cmd_spec:
                        cmds.remove(element)
                        cmd_spec.remove(element)

            if options:
                continue  # not all options have been matched
            if cmds:
                continue  # not all cmds from command line have been matched
            if args_spec:
                continue  # not all args from spec have been provided
            if cmd_spec:
                continue  # not all cmds from spec have been provided
            # all options and cmds matched : call the cmd
            # TODO leave out all args to deal with "empty" signature
            exit_code = func(*args, **kwargs)
            return exit_code
        # no matching spec found
        raise Exception('No implementation for spec: %s' % arguments)