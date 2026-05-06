def wrapped_sendfile(act, offset, length):
    """
    Calls the sendfile system call or simulate with file read and socket send if
    unavailable.
    """
    if sendfile:
        offset, sent = sendfile.sendfile(
            act.sock.fileno(),
            act.file_handle.fileno(),
            offset, length
        )
    else:
        act.file_handle.seek(offset)
        sent = act.sock._fd.send(act.file_handle.read(length))
    return sent