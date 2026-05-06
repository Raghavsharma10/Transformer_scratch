def _subcommand_arguments(args):
    """
    Return (subcommand, (possibly adjusted) arguments for that subcommand)

    Returns (None, args) when no subcommand is found

    Parsing our arguments is hard. Each subcommand has its own docopt
    validation, and some subcommands (paster and shell) have positional
    options (some options passed to datacats and others passed to
    commands run inside the container)
    """
    skip_site = False
    # Find subcommand without docopt so that subcommand options may appear
    # anywhere
    for i, a in enumerate(args):
        if skip_site:
            skip_site = False
            continue
        if a.startswith('-'):
            if a == '-s' or a == '--site':
                skip_site = True
            continue
        if a == 'help':
            return _subcommand_arguments(args[:i] + ['--help'] + args[i + 1:])
        if a not in COMMANDS:
            raise DatacatsError("\'{0}\' command is not recognized. \n"
              "See \'datacats help\' for the list of available commands".format(a))
        command = a
        break
    else:
        return None, args

    if command != 'shell' and command != 'paster':
        return command, args

    # shell requires the environment name, paster does not
    remaining_positional = 2 if command == 'shell' else 1

    # i is where the subcommand starts.
    # shell, paster are special: options might belong to the command being
    # find where the the inner command starts and insert a '--' before
    # so that we can separate inner options from ones we need to parse
    while i < len(args):
        a = args[i]
        if a.startswith('-'):
            if a == '-s' or a == '--site':
                # site name is coming
                i += 2
                continue
            i += 1
            continue
        if remaining_positional:
            remaining_positional -= 1
            i += 1
            continue
        return command, args[:i] + ['--'] + args[i:]

    return command, args