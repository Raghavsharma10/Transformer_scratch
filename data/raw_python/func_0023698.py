def execute_show(args, root_dir):
    """Print stderr and stdout of the current running process.

    Args:
        args['watch'] (bool): If True, we open a curses session and tail
                              the output live in the console.
        root_dir (string): The path to the root directory the daemon is running in.

    """
    key = None
    if args.get('key'):
        key = args['key']
        status = command_factory('status')({}, root_dir=root_dir)
        if key not in status['data'] or status['data'][key]['status'] != 'running':
            print('No running process with this key, use `log` to show finished processes.')
            return

    # In case no key provided, we take the oldest running process
    else:
        status = command_factory('status')({}, root_dir=root_dir)
        if isinstance(status['data'], str):
            print(status['data'])
            return
        for k in sorted(status['data'].keys()):
            if status['data'][k]['status'] == 'running':
                key = k
                break
        if key is None:
            print('No running process, use `log` to show finished processes.')
            return

    config_dir = os.path.join(root_dir, '.config/pueue')
    # Get current pueueSTDout file from tmp
    stdoutFile = os.path.join(config_dir, 'pueue_process_{}.stdout'.format(key))
    stderrFile = os.path.join(config_dir, 'pueue_process_{}.stderr'.format(key))
    stdoutDescriptor = open(stdoutFile, 'r')
    stderrDescriptor = open(stderrFile, 'r')
    running = True
    # Continually print output with curses or just print once
    if args['watch']:
        # Initialize curses
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(2)
        stdscr.keypad(True)
        stdscr.refresh()

        try:
            # Update output every two seconds
            while running:
                stdscr.clear()
                stdoutDescriptor.seek(0)
                message = stdoutDescriptor.read()
                stdscr.addstr(0, 0, message)
                stdscr.refresh()
                time.sleep(2)
        except Exception:
            # Curses cleanup
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()
    else:
        print('Stdout output:\n')
        stdoutDescriptor.seek(0)
        print(get_descriptor_output(stdoutDescriptor, key))
        print('\n\nStderr output:\n')
        stderrDescriptor.seek(0)
        print(get_descriptor_output(stderrDescriptor, key))