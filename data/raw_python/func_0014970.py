def init():
    ''' Initialize the library '''
    globals()["sock"] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    globals()["sockfd"] = globals()["sock"].fileno()