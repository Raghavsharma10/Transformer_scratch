def get_my_ips():
    """highly os specific - works only in modern linux kernels"""
    ips = list()
    if not os.path.exists("/sys/class/net"): # not linux
        return ['127.0.0.1']
    for ifdev in os.listdir("/sys/class/net"):
        if ifdev == "lo":
            continue
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            ips.append(socket.inet_ntoa(fcntl.ioctl(
                sock.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack('256s', ifdev[:15].encode())
            )[20:24]))
        except OSError:
            pass
    return ips