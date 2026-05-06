def manage(commands, argv=None, delim=':'):
    '''
    Parses argv and runs neccessary command. Is to be used in manage.py file.

    Accept a dict with digest name as keys and instances of
    :class:`Cli<iktomi.management.commands.Cli>`
    objects as values.

    The format of command is the following::

        ./manage.py digest_name:command_name[ arg1[ arg2[...]]][ --key1=kwarg1[...]]

    where command_name is a part of digest instance method name, args and kwargs
    are passed to the method. For details, see
    :class:`Cli<iktomi.management.commands.Cli>` docs.
    '''

    commands = {(k.decode('utf-8') if isinstance(k, six.binary_type) else k): v
                for k, v in commands.items()}

    # Default django autocompletion script is registered to manage.py
    # We use the same name for this script and it seems to be ok
    # to implement the same interface
    def perform_auto_complete(commands):
        from .lazy import LazyCli
        cwords = os.environ['COMP_WORDS'].split()[1:]
        cword = int(os.environ['COMP_CWORD'])

        try:
            curr = cwords[cword - 1]
        except IndexError:
            curr = ''

        suggest = []
        if len(cwords) > 1 and cwords[0] in commands.keys():
            value = commands[cwords[0]]
            if isinstance(value, LazyCli):
                value = value.get_digest()

            for cmd_name, _ in value.get_funcs():
                cmd_name = cmd_name[8:]
                suggest.append(cmd_name)
            if curr == ":":
                curr = ''
        else:
            suggest += list(commands.keys()) + [x+":" for x in commands.keys()]
        suggest.sort()
        output = u" ".join(filter(lambda x: x.startswith(curr), suggest))
        sys.stdout.write(output)

    auto_complete = 'IKTOMI_AUTO_COMPLETE' in os.environ or \
                    'DJANGO_AUTO_COMPLETE' in os.environ
    if auto_complete:
        perform_auto_complete(commands)
        sys.exit(0)

    argv = sys.argv if argv is None else argv
    if len(argv) > 1:
        cmd_name = argv[1]
        raw_args = argv[2:]
        args, kwargs = [], {}
        # parsing params
        for item in raw_args:
            if item.startswith('--'):
                splited = item[2:].split('=', 1)
                if len(splited) == 2:
                    k,v = splited
                elif len(splited) == 1:
                    k,v = splited[0], True
                kwargs[k] = v
            else:
                args.append(item)

        # trying to get command instance
        if delim in cmd_name:
            digest_name, command = cmd_name.split(delim)
        else:
            digest_name = cmd_name
            command = None
        try:
            digest = commands[digest_name]
        except KeyError:
            _command_list(commands)
            sys.exit('ERROR: Command "{}" not found'.format(digest_name))
        try:
            if command is None:
                if isinstance(digest, Cli):
                    help_ = digest.description(argv[0], digest_name)
                    sys.stdout.write(help_)
                    sys.exit('ERROR: "{}" command digest requires command name'\
                                .format(digest_name))
                digest(*args, **kwargs)
            else:
                digest(command, *args, **kwargs)
        except CommandNotFound:
            help_ = digest.description(argv[0], digest_name)
            sys.stdout.write(help_)
            sys.exit('ERROR: Command "{}:{}" not found'.format(digest_name, command))
    else:
        _command_list(commands)
        sys.exit('Please provide any command')