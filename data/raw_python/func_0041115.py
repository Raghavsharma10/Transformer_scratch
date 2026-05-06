def socket_set_hwm(socket, hwm=-1):
    """libzmq 2/3 compatible sethwm"""
    try:
        socket.sndhwm = socket.rcvhwm = hwm
    except AttributeError:
        socket.hwm = hwm