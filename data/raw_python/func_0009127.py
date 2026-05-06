def remote_pdb_handler(signum, frame):
    """ Handler to drop us into a remote debugger upon receiving SIGUSR1 """
    try:
        from remote_pdb import RemotePdb

        rdb = RemotePdb(host="127.0.0.1", port=0)
        rdb.set_trace(frame=frame)
    except ImportError:
        log.warning(
            "remote_pdb unavailable.  Please install remote_pdb to "
            "allow remote debugging."
        )
    # Restore signal handler for later
    signal.signal(signum, remote_pdb_handler)