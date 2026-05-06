def _mksocket(host, port, q, done, stop):
    """Returns a tcp socket to (host/port). Retries forever if connection fails"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    attempt = 0
    while not stop.is_set():
        try:
            s.connect((host, port))
            return s
        except Exception as ex:
            # Simple exponential backoff: sleep for 1,2,4,8,16,30,30...
            time.sleep(min(30, 2 ** attempt))
            attempt += 1