def sam_readline(sock, partial = None):
    """read a line from a sam control socket"""
    response = b''
    exception = None
    while True:
        try:
            c = sock.recv(1)
            if not c:
                raise EOFError('SAM connection died. Partial response %r %r' % (partial, response))
            elif c == b'\n':
                break
            else:
                response += c
        except (BlockingIOError, pysocket.timeout) as e:
            if partial is None:
                raise e
            else:
                exception = e
                break

    if partial is None:
        # print('<--', response)
        return response.decode('ascii')
    else:
        # print('<--', repr(partial), '+', response, exception)
        return (partial + response.decode('ascii'), exception)