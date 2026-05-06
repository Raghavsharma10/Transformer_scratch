def write(bar, offset, data):
    """Write data to PCI board.
    
    Parameters
    ----------
    bar : BaseAddressRegister
        BAR to write.
    
    offset : int
        Address offset in BAR to write.
        
    data : bytes
        Data to write.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> b = pypci.lspci(vendor=0x1147, device=3214)
    
    >>> pypci.write(b[0].bar[2], 0x04, b'\x01')
    
    >>> data = struct.pack('<I', 1234567)
    >>> pypci.write(b[0].bar[2], 0x00, data)
    """
    if type(data) not in [bytes, bytearray]:
        msg = 'data should be bytes or bytearray type'
        raise TypeError(msg)
    
    size = len(data)
    verify_access_range(bar, offset, size)
    if bar.type == 'io': return io_write(bar, offset, data)
    if bar.type == 'mem': return mem_write(bar, offset, data)
    return