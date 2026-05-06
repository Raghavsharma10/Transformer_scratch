def list_codes(remote, device=None, address=None):
    """
    List the codes for a given remote.

    All parameters are passed to irsend. See the man page for irsend
    for details about their usage.

    Parameters
    ----------
    remote: str
    device: str
    address: str

    Returns
    -------
    [str]

    Notes
    -----
    No attempt is made to catch or handle errors. See the documentation
    for subprocess.check_output to see the types of exceptions it may raise.

    """
    output = _call(["list", remote, ""], None, device, address)
    codes = [l.split()[-1] for l in output.splitlines() if l]
    return codes