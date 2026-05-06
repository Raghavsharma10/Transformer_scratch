def set_terminate_listeners(stream):
    """Die on SIGTERM or SIGINT"""

    def stop(signum, frame):
        terminate(stream.listener)

    # Installs signal handlers for handling SIGINT and SIGTERM
    # gracefully.
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)