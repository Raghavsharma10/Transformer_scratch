def getLocalIPaddress():
    """visible to other machines on LAN"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('google.com', 0))
        my_local_ip = s.getsockname()[0] # takes ~0.005s
        #from netifaces import interfaces, ifaddresses, AF_INET
        #full solution in the event of multiple NICs (network interface cards) on the PC
        #def ip4_addresses():
        #    ip_list = []
        #    for interface in interfaces():
        #        for link in ifaddresses(interface)[AF_INET]: # If IPv6 addresses are needed instead, use AF_INET6 instead of AF_INET
        #            ip_list.append(link['addr'])
        #    return ip_list
    except Exception:
        my_local_ip = None
    return my_local_ip