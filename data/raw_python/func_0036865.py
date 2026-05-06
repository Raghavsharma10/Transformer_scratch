def lspci(vendor=None, device=None):
    """Collect PCI information and return its list.
    
    Parameters
    ----------
    vendor : int
        Return only devices with specified vendor ID.
    
    device : int
        Return only devices with specified device ID.
    
    Returns
    -------
    list
        List of PCI device information.
        Information are stored in PCIConfigHeader namedtuple object.
    
    Examples
    --------
    >>> b = pypci.lspci(vendor=0x1147, device=3214)
    
    >>> b[0].vendor_id
    4423
    
    >>> b[0].bar
    [BaseAddressRegister(type='mem', addr=2421170176, size=64),
     BaseAddressRegister(type='mem', addr=2421166080, size=64),
     BaseAddressRegister(type='mem', addr=2421174272, size=32)]
    """
    lspci_cmd = ['lspci', '-xxxx', '-v']
    lspci_results = subprocess.run(lspci_cmd, stdout=subprocess.PIPE)
    lspci_stdout = lspci_results.stdout.decode('utf-8')
    config = [parse_lspci_output(stdout) for stdout
              in lspci_stdout.split('\n\n') if stdout != '']
    
    if vendor is not None:
        config = [c for c in config if c.vendor_id==vendor]
        pass
    
    if device is not None:
        config = [c for c in config if c.device_id==device]
        pass
    
    return config