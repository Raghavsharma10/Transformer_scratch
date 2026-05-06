def main():
    """Execute entry function."""
    args = parser.parse_args()
    args_dict = vars(args)
    root_dir = args_dict['root'] if 'root' in args else None

    # If a root directory is specified, get the absolute path and
    # check if it exists. Abort if it doesn't exist!
    if root_dir:
        root_dir = os.path.abspath(root_dir)
        if not os.path.exists(root_dir):
            print("The specified directory doesn't exist!")
            sys.exit(1)
    # Default to home directory if no root is specified
    else:
        root_dir = os.path.expanduser('~')

    if args.stopdaemon:
        print_command_factory('STOPDAEMON')(vars(args), root_dir)
    elif args.nodaemon:
        daemon_factory(root_dir)()
    elif args.daemon:
        config_dir = os.path.join(root_dir, '.config/pueue')
        os.makedirs(config_dir, exist_ok=True)
        daemon = Daemonize(app='pueue', pid=os.path.join(config_dir, 'pueue.pid'),
                           action=daemon_factory(root_dir), chdir=root_dir)
        daemon.start()
    elif hasattr(args, 'func'):
        try:
            args.func(args_dict, root_dir)
        except EOFError:
            print('Apparently the daemon just died. Sorry for that :/')
    else:
        print('Invalid Command. Please check -h')