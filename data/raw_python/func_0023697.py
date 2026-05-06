def execute_log(args, root_dir):
    """Print the current log file.

    Args:
        args['keys'] (int): If given, we only look at the specified processes.
        root_dir (string): The path to the root directory the daemon is running in.

    """
    # Print the logs of all specified processes
    if args.get('keys'):
        config_dir = os.path.join(root_dir, '.config/pueue')
        queue_path = os.path.join(config_dir, 'queue')
        if os.path.exists(queue_path):
            queue_file = open(queue_path, 'rb')
            try:
                queue = pickle.load(queue_file)
            except Exception:
                print('Queue log file seems to be corrupted. Aborting.')
                return
            queue_file.close()
        else:
            print('There is no queue log file. Aborting.')
            return

        for key in args.get('keys'):
            # Check if there is an entry with this key
            if queue.get(key) and queue[key]['status'] in ['failed', 'done']:
                entry = queue[key]
                print('Log of entry: {}'.format(key))
                print('Returncode: {}'.format(entry['returncode']))
                print('Command: {}'.format(entry['command']))
                print('Path: {}'.format(entry['path']))
                print('Start: {}, End: {} \n'.format(entry['start'], entry['end']))

                # Write STDERR
                if len(entry['stderr']) > 0:
                    print(Color('{autored}Stderr output: {/autored}\n    ') + entry['stderr'])

                # Write STDOUT
                if len(entry['stdout']) > 0:
                    print(Color('{autogreen}Stdout output: {/autogreen}\n    ') + entry['stdout'])
            else:
                print('No finished process with key {}.'.format(key))

    # Print the log of all processes
    else:
        log_path = os.path.join(root_dir, '.local/share/pueue/queue.log')
        log_file = open(log_path, 'r')
        print(log_file.read())