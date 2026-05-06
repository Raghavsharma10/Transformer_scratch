def getLTime():
    """Returns a formatted string with the current local time."""

    _ltime = _time.localtime(_time.time())
    tlm_str = _time.strftime('%H:%M:%S (%d/%m/%Y)', _ltime)
    return tlm_str