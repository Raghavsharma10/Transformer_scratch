def execute_add(args, root_dir=None):
    """Add a new command to the daemon queue.

    Args:
        args['command'] (list(str)): The actual programm call. Something like ['ls', '-a'] or ['ls -al']
        root_dir (string): The path to the root directory the daemon is running in.
    """

    # We accept a list of strings.
    # This is done to create a better commandline experience with argparse.
    command = ' '.join(args['command'])

    # Send new instruction to daemon
    instruction = {
        'command': command,
        'path': os.getcwd()
    }
    print_command_factory('add')(instruction, root_dir)