def register(host=DFLT_ADDRESS[0], port=DFLT_ADDRESS[1],
             signum=signal.SIGUSR1):
    """Register a pdb handler for signal 'signum'.

    The handler sets pdb to listen on the ('host', 'port') internet address
    and to start a remote debugging session on accepting a socket connection.
    """
    _pdbhandler._register(host, port, signum)