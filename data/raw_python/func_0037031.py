def init():
    """Initialization , all begin from here
    """
    su()
    args = sys.argv
    args.pop(0)
    cmd = "{0}sun_daemon".format(bin_path)
    if len(args) == 1:
        if args[0] == "start":
            print("Starting SUN daemon:  {0} &".format(cmd))
            subprocess.call("{0} &".format(cmd), shell=True)
        elif args[0] == "stop":
            print("Stopping SUN daemon:  {0}".format(cmd))
            subprocess.call("killall sun_daemon", shell=True)
        elif args[0] == "restart":
            print("Stopping SUN daemon:  {0}".format(cmd))
            subprocess.call("killall sun_daemon", shell=True)
            print("Starting SUN daemon:  {0} &".format(cmd))
            subprocess.call("{0} &".format(cmd), shell=True)
        elif args[0] == "check":
            _init_check_upodates()
        elif args[0] == "status":
            print(daemon_status())
        elif args[0] == "help":
            usage()
        elif args[0] == "info":
            print(os_info())
        else:
            print("try: 'sun help'")
    elif len(args) == 2 and args[0] == "start" and args[1] == "--gtk":
        subprocess.call("{0} {1}".format(cmd, "start--gtk"), shell=True)
    else:
        print("try: 'sun help'")