def find_device(vidpid):
    """Finds a connected device with the given VID:PID. Returns the serial
    port url."""
    for port in list_ports.comports():
        if re.search(vidpid, port[2], flags=re.IGNORECASE):
            return port[0]

    raise exceptions.RoasterLookupError