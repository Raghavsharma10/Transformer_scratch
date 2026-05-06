def start_daemon_thread(target, args=()):
    """starts a deamon thread for a given target function and arguments."""
    th = Thread(target=target, args=args)
    th.daemon = True
    th.start()
    return th