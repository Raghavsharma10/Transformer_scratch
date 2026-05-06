def stop_main_thread(*args):
    """
    CLEAN OF ALL THREADS CREATED WITH THIS LIBRARY
    """
    try:
        if len(args) and args[0] != _signal.SIGTERM:
            Log.warning("exit with {{value}}", value=_describe_exit_codes.get(args[0], args[0]))
    except Exception as _:
        pass
    finally:
        MAIN_THREAD.stop()