def watch(directory=None, auto_clear=False, extensions=[]):
    """Starts a server to render the specified file or directory containing a README."""
    if directory and not os.path.isdir(directory):
        raise ValueError('Directory not found: ' + directory)
    directory = os.path.abspath(directory)

    # Initial run
    event_handler = ChangeHandler(directory, auto_clear, extensions)
    event_handler.run()

    # Setup watchdog
    observer = Observer()
    observer.schedule(event_handler, path=directory, recursive=True)
    observer.start()

    # Watch and run tests until interrupted by user
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()