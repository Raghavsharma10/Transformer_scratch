def discover_upnp_devices(
        self, st="upnp:rootdevice", timeout=2, mx=1, retries=1
    ):
        """
        sends an SSDP discovery packet to the network and collects
        the devices that replies to it. A dictionary is returned
        using the devices unique usn as key
        """
        # prepare UDP socket to transfer the SSDP packets
        s = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        s.settimeout(timeout)

        # prepare SSDP discover message
        msg = SSDPDiscoveryMessage(mx=mx, st=st)

        # try to get devices with multiple retries in case of failure
        devices = {}
        for _ in range(retries):
            # send SSDP discovery message
            s.sendto(msg.bytes, SSDP_MULTICAST_ADDR)

            devices = {}
            try:
                while True:
                    # parse response and store it in dict
                    r = SSDPResponse(s.recvfrom(65507))
                    devices[r.usn] = r

            except socket.timeout:
                break

        return devices