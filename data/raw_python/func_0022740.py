def _stdin_ready_posix():
    """Return True if there's something to read on stdin (posix version)."""
    infds, outfds, erfds = select.select([sys.stdin],[],[],0)
    return bool(infds)