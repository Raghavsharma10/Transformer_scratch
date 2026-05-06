def run_alias():
    """
    Quick aliases for run command.
    """
    mode = Path(sys.argv[0]).stem
    help = True if len(sys.argv) <= 1 else False
    if mode == 'lcc':
        sys.argv.insert(1, 'c')
    elif mode == 'lpython':
        sys.argv.insert(1, 'python')
    sys.argv.insert(1, 'run')
    if help:
        sys.argv.append('--help')
    main.main(prog_name='backend.ai')