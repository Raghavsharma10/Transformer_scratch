def reload(*command, ignore_patterns=[]):
    """Reload given command"""
    path = "."
    sig = signal.SIGTERM
    delay = 0.25
    ignorefile = ".reloadignore"

    ignore_patterns = ignore_patterns or load_ignore_patterns(ignorefile)

    event_handler = ReloadEventHandler(ignore_patterns)
    reloader = Reloader(command, signal)

    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    reloader.start_command()

    try:
        while True:
            time.sleep(delay)
            sys.stdout.write(reloader.read())
            sys.stdout.flush()
            if event_handler.modified:
                reloader.restart_command()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    reloader.stop_command()
    sys.stdout.write(reloader.read())
    sys.stdout.flush()