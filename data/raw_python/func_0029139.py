def start(controller_class):
    """Start the Helper controller either in the foreground or as a daemon
    process.

    :param controller_class: The controller class handle to create and run
    :type controller_class: callable

    """
    args = parser.parse()
    obj = controller_class(args, platform.operating_system())
    if args.foreground:
        try:
            obj.start()
        except KeyboardInterrupt:
            obj.stop()
    else:
        try:
            with platform.Daemon(obj) as daemon:
                daemon.start()
        except (OSError, ValueError) as error:
            sys.stderr.write('\nError starting %s: %s\n\n' %
                             (sys.argv[0], error))
            sys.exit(1)