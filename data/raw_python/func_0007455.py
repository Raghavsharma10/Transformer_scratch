def _update_project_watch(config, task_presenter, results,
                          long_description, tutorial):  # pragma: no cover
    """Update a project in a loop."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = os.getcwd()
    event_handler = PbsHandler(config, task_presenter, results,
                               long_description, tutorial)
    observer = Observer()
    # We only want the current folder, not sub-folders
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()