def autocomplete():
    """Command and option completion for the main option parser (and options)
    and its subcommands (and options).

    Enable by sourcing one of the completion shell scripts (bash or zsh).
    """
    # Don't complete if user hasn't sourced bash_completion file.
    if 'PIP_AUTO_COMPLETE' not in os.environ:
        return
    cwords = os.environ['COMP_WORDS'].split()[1:]
    cword = int(os.environ['COMP_CWORD'])
    try:
        current = cwords[cword-1]
    except IndexError:
        current = ''
    load_all_commands()
    subcommands = [cmd for cmd, cls in command_dict.items() if not cls.hidden]
    options = []
    # subcommand
    try:
        subcommand_name = [w for w in cwords if w in subcommands][0]
    except IndexError:
        subcommand_name = None
    # subcommand options
    if subcommand_name:
        # special case: 'help' subcommand has no options
        if subcommand_name == 'help':
            sys.exit(1)
        # special case: list locally installed dists for uninstall command
        if subcommand_name == 'uninstall' and not current.startswith('-'):
            installed = []
            lc = current.lower()
            for dist in get_installed_distributions(local_only=True):
                if dist.key.startswith(lc) and dist.key not in cwords[1:]:
                    installed.append(dist.key)
            # if there are no dists installed, fall back to option completion
            if installed:
                for dist in installed:
                    print(dist)
                sys.exit(1)
        subcommand = command_dict.get(subcommand_name)
        options += [(opt.get_opt_string(), opt.nargs)
                    for opt in subcommand.parser.option_list
                    if opt.help != optparse.SUPPRESS_HELP]
        # filter out previously specified options from available options
        prev_opts = [x.split('=')[0] for x in cwords[1:cword-1]]
        options = [(x, v) for (x, v) in options if x not in prev_opts]
        # filter options by current input
        options = [(k, v) for k, v in options if k.startswith(current)]
        for option in options:
            opt_label = option[0]
            # append '=' to options which require args
            if option[1]:
                opt_label += '='
            print(opt_label)
    else:
        # show options of main parser only when necessary
        if current.startswith('-') or current.startswith('--'):
            subcommands += [opt.get_opt_string()
                            for opt in parser.option_list
                            if opt.help != optparse.SUPPRESS_HELP]
        print(' '.join([x for x in subcommands if x.startswith(current)]))
    sys.exit(1)