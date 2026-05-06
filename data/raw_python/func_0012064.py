def main():
    """
    parse command line options and either launch some configuration dialog or start an instance of _MainLoop as a daemon
    """
    (options, _) = _parse_args()

    if options.change_password:
        c.keyring_set_password(c["username"])
        sys.exit(0)

    if options.select:
        courses = client.get_courses()
        c.selection_dialog(courses)
        c.save()
        sys.exit(0)

    if options.stop:
        os.system("kill -2 `cat ~/.studdp/studdp.pid`")
        sys.exit(0)

    task = _MainLoop(options.daemonize, options.update_courses)

    if options.daemonize:
        log.info("daemonizing...")
        with daemon.DaemonContext(working_directory=".", pidfile=PIDLockFile(PID_FILE)):
            # we have to create a new logger in the daemon context
            handler = logging.FileHandler(LOG_PATH)
            handler.setFormatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            log.addHandler(handler)
            task()
    else:
        task()