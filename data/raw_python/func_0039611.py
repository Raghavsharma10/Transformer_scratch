def getSystemIps():
        """ will not return the localhost one """
        IPs = []
        for interface in NetInfo.getSystemIfs():
            if not interface.startswith('lo'):
                ip = netinfo.get_ip(interface)
                IPs.append(ip)
        return IPs