def connect(self, attempts=20, delay=0.5):
        """
        Connects to a gateway, blocking until a connection is made and bulbs
        are found.

        Step 1: send a gateway discovery packet to the broadcast address, wait
        until we've received some info about the gateway.

        Step 2: connect to a discovered gateway, wait until the connection has
        been completed.

        Step 3: ask for info about bulbs, wait until we've found the number of
        bulbs we expect.

        Raises a ConnectException if any of the steps fail.
        """
        # Broadcast discovery packets until we find a gateway.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        with closing(sock):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            discover_packet = build_packet(REQ_GATEWAY,
                                           ALL_BULBS, ALL_BULBS, '',
                                           protocol=DISCOVERY_PROTOCOL)

            for _, ok in _retry(self.gateway_found_event, attempts, delay):
                sock.sendto(discover_packet, BROADCAST_ADDRESS)
        if not ok:
            raise ConnectException('discovery failed')
        self.callbacks.put(EVENT_DISCOVERED)

        # Tell the sender to connect to the gateway until it does.
        for _, ok in _retry(self.sender.is_connected, 1, 3):
            self.sender.put(self.gateway)
        if not ok:
            raise ConnectException('connection failed')
        self.callbacks.put(EVENT_CONNECTED)

        # Send light state packets to the gateway until we find bulbs.
        for _, ok in _retry(self.bulbs_found_event, attempts, delay):
            self.send(REQ_GET_LIGHT_STATE, ALL_BULBS, '')
        if not ok:
            raise ConnectException('only found %d of %d bulbs' % (
                                   len(self.bulbs), self.num_bulbs))
        self.callbacks.put(EVENT_BULBS_FOUND)