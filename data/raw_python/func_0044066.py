def sam_send(sock, line_and_data):
    """Send a line to the SAM controller, but don't read it"""
    if isinstance(line_and_data, tuple):
        line, data = line_and_data
    else:
        line, data = line_and_data, b''

    line = bytes(line, encoding='ascii') + b' \n'
    # print('-->', line, data)
    sock.sendall(line + data)