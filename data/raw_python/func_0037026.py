def send_start(remote, code, device=None, address=None):
    """
    All parameters are passed to irsend. See the man page for irsend
    for details about their usage.

    Parameters
    ----------
    remote: str
    code: str
    device: str
    address: str

    Notes
    -----
    No attempt is made to catch or handle errors. See the documentation
    for subprocess.check_output to see the types of exceptions it may raise.

    """
    args = ['send_start', remote, code]
    _call(args, device, address)