def run_workers(no_subprocess, watch_paths=None, is_background=False):
    """
    subprocess handler
    """
    import atexit, os, subprocess, signal
    if watch_paths:
        from watchdog.observers import Observer
        # from watchdog.observers.fsevents import FSEventsObserver as Observer
        # from watchdog.observers.polling import PollingObserver as Observer
        from watchdog.events import FileSystemEventHandler

    def on_modified(event):
        if not is_background:
            print("Restarting worker due to change in %s" % event.src_path)
        log.info("modified %s" % event.src_path)
        try:
            kill_children()
            run_children()
        except:
            log.exception("Error while restarting worker")

    handler = FileSystemEventHandler()
    handler.on_modified = on_modified

    # global child_pids
    child_pids = []
    log.info("starting %s workers" % no_subprocess)

    def run_children():
        global child_pids
        child_pids = []
        for i in range(int(no_subprocess)):
            proc = subprocess.Popen([sys.executable, __file__],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            child_pids.append(proc.pid)
            log.info("Started worker with pid %s" % proc.pid)

    def kill_children():
        """
        kill subprocess on exit of manager (this) process
        """
        log.info("Stopping worker(s)")
        for pid in child_pids:
            if pid is not None:
                os.kill(pid, signal.SIGTERM)

    run_children()
    atexit.register(kill_children)
    signal.signal(signal.SIGTERM, kill_children)
    if watch_paths:
        observer = Observer()
        for path in watch_paths:
            if not is_background:
                print("Watching for changes under %s" % path)
            observer.schedule(handler, path=path, recursive=True)
        observer.start()
    while 1:
        try:
            sleep(1)
        except KeyboardInterrupt:
            log.info("Keyboard interrupt, exiting")
            if watch_paths:
                observer.stop()
                observer.join()
            sys.exit(0)