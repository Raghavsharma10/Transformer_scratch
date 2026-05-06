def hardware_info():
    """
    Returns basic hardware information about the computer.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on.

    Returns
    -------
    info : dict
        Dictionary containing cpu and memory information.

    """
    try:
        if sys.platform == 'darwin':
            out = _mac_hardware_info()
        elif sys.platform == 'win32':
            out = _win_hardware_info()
        elif sys.platform in ['linux', 'linux2']:
            out = _linux_hardware_info()
        else:
            out = {}
    except:
        return {}
    else:
        return out