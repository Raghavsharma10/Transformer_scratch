def set_transmitters(transmitters, device=None, address=None):
    """
    All parameters are passed to irsend. See the man page for irsend
    for details about their usage.

    Parameters
    ----------
    transmitters: iterable yielding ints
    device: str
    address: str

    Notes
    -----
    No attempt is made to catch or handle errors. See the documentation
    for subprocess.check_output to see the types of exceptions it may raise.

    """
    args = ['set_transmitters'] + [str(i) for i in transmitters]
    _call(args, None, device, address)