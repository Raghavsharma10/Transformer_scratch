def receive_data(socket):
    """Receive an answer from the daemon and return the response.

    Args:
    socket (socket.socket): A socket that is connected to the daemon.

    Returns:
        dir or string: The unpickled answer.
    """
    answer = b""
    while True:
        packet = socket.recv(4096)
        if not packet: break
        answer += packet
    response = pickle.loads(answer)
    socket.close()
    return response