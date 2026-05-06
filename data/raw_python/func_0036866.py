def read(bar, offset, size):
    """Read PCI data register.
    
    Parameters
    ----------
    bar : BaseAddressRegister
        BAR to read.
    
    offset : int
        Address offset in BAR to read.
    
    size : int
        Data size to read.
        
    Returns
    -------
    bytes
        Data read from BAR
        
    Examples
    --------
    >>> b = pypci.lspci(vendor=0x1147, device=3214)
    >>> pypci.read(b[0].bar[2], 0x0c, 4)
    b'\x00\x00\x00\x0c'
    """
    verify_access_range(bar, offset, size)
    if bar.type == 'io': return io_read(bar, offset, size)
    if bar.type == 'mem': return mem_read(bar, offset, size)
    return b''