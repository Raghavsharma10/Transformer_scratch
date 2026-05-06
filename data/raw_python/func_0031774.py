def watch_record(indexer, use_polling=False):
    """
    Start watching `cfstore.record_path`.

    :type indexer: rash.indexer.Indexer

    """
    if use_polling:
        from watchdog.observers.polling import PollingObserver as Observer
        Observer  # fool pyflakes
    else:
        from watchdog.observers import Observer

    event_handler = RecordHandler(indexer)
    observer = Observer()
    observer.schedule(event_handler, path=indexer.record_path, recursive=True)
    indexer.logger.debug('Start observer.')
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        indexer.logger.debug('Got KeyboardInterrupt. Stopping observer.')
        observer.stop()
    indexer.logger.debug('Joining observer.')
    observer.join()
    indexer.logger.debug('Finish watching record.')