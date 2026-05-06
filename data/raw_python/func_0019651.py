def socket_read(fp):
    """Buffered read from socket. Reads all data available from socket.
    
    @fp:     File pointer for socket.
    @return: String of characters read from buffer.
    
    """
    response = ''
    oldlen = 0
    newlen = 0
    while True:
        response += fp.read(buffSize)
        newlen = len(response)
        if newlen - oldlen == 0:
            break
        else:
            oldlen = newlen
    return response