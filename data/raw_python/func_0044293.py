def shutdown(server, graceful=True):
    """Shut down the application.

    If a graceful stop is requested, waits for all of the IO loop's
    handlers to finish before shutting down the rest of the process.
    We impose a 10 second timeout.

    Based on http://tornadogists.org/3428652/
    """
    ioloop = IOLoop.instance()

    logging.info("Stopping server...")
    # Stop listening for new connections
    server.stop()

    def final_stop():
        ioloop.stop()
        logging.info("Stopped.")
        sys.exit(0)

    def poll_stop(counts={'remaining': None, 'previous': None}):
        remaining = len(ioloop._handlers)
        counts['remaining'], counts['previous'] = remaining, counts['remaining']
        previous = counts['previous']
        # Wait until we only have only one IO handler remaining.  That
        # final handler will be our PeriodicCallback polling task.
        if remaining == 1:
            final_stop()
        if previous is None or remaining != previous:
            logging.info("Waiting on IO %d remaining handlers", remaining)

    if graceful:
        # Callback to check on remaining handlers.
        poller = PeriodicCallback(poll_stop, 250, io_loop=ioloop)
        poller.start()

        # Give up after 10 seconds of waiting.
        ioloop.add_timeout(time.time() + 10, final_stop)
    else:
        final_stop()