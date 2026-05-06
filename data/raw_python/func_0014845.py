def at(host, command, seq, params):
    """
    Parameters:
    command -- the command
    seq -- the sequence number
    params -- a list of elements which can be either int, float or string
    """
    params_str = []
    for p in params:
        if type(p) == int:
            params_str.append('{:d}'.format(p))
        elif type(p) == float:
            params_str.append('{:d}'.format(f2i(p)))
        elif type(p) == str:
            params_str.append('"{:s}"'.format(p))
    msg = 'AT*{:s}={:d},{:s}\r'.format(command, seq, ','.join(params_str))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(msg.encode(), (host, ardrone.constant.COMMAND_PORT))