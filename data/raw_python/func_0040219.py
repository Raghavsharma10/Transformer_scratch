def daemon_start(main, pidfile, daemon=True, workspace=None):
    """Start application in background mode if required and available. If not then in front mode.
    """
    logger.debug("start daemon application pidfile={pidfile} daemon={daemon} workspace={workspace}.".format(pidfile=pidfile, daemon=daemon, workspace=workspace))
    new_pid = os.getpid()
    workspace = workspace or os.getcwd()
    os.chdir(workspace)
    daemon_flag = False
    if pidfile and daemon:
        old_pid = load_pid(pidfile)
        if old_pid:
            logger.debug("pidfile {pidfile} already exists, pid={pid}.".format(pidfile=pidfile, pid=old_pid))
        # if old service is running, just exit.
        if old_pid and is_running(old_pid):
            error_message = "Service is running in process: {pid}.".format(pid=old_pid)
            logger.error(error_message)
            six.print_(error_message, file=os.sys.stderr)
            os.sys.exit(95)
        # clean old pid file.
        clean_pid_file(pidfile)
        # start as background mode if required and available.
        if daemon and os.name == "posix":
            make_basic_daemon()
            daemon_flag = True
    if daemon_flag:
        logger.info("Start application in DAEMON mode, pidfile={pidfile} pid={pid}".format(pidfile=pidfile, pid=new_pid))
    else:
        logger.info("Start application in FRONT mode, pid={pid}.".format(pid=new_pid))
    write_pidfile(pidfile)
    atexit.register(clean_pid_file, pidfile)
    main()
    return